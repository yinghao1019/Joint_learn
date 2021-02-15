import os
import logging
import numpy as np
import random
import tqdm
import utils
import json

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from utils import MODEL_CLASSES, MODEL_PATH, get_intent_labels, get_slot_labels, get_modelMetrics, count_modelParams
from transformers import AdamW, get_linear_schedule_with_warmup, PreTrainedModel


logger = logging.getLogger(__name__)


class BaseTrainer:
    def __init__(self, train_set, val_set, test_set, args, model=None, pretrained_path=None):

        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set
        self.args = args
        self.word_vocab = utils.get_word_vocab(args)
        self.intent_vocab = get_intent_labels(args)
        self.slot_vocab = get_slot_labels(args)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() \
            and not args.no_cuda else torch.device('cpu')
        self.model = model

    def train_model(self):
        # build train iterator
        sampler = RandomSampler(self.train_set)
        data_iter = DataLoader(self.train_set, self.args.bs, sampler)

        # compute train step
        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = t_total//(
                len(data_iter)//self.args.grad_accumulate_steps)
        else:
            t_total = self.args.num_train_epochs * \
                (len(data_iter)//self.args.grad_accumulate_steps)
        # prepare lr scheduler and optimizer
        no_decay = ['LayerNorm', 'bias']
        param_gropus = [
            {'params': [p for n, p in self.model.named_parameters() if not any([nd in n for nd in no_decay])],
             'weight_decay':self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any([nd in n for nd in no_decay])],
             'weight_decay':0.0},
        ]

        optimizer = AdamW(param_gropus, lr=self.args.train_lr,
                          eps=self.args.adam_epsilon)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, self.args.warm_steps, t_total)

        # build train_progress
        train_pgb = tqdm.trange(self.args.num_train_epochs, desc='EPOCHS')
        global_steps = 0
        total_loss = 0

        # count params
        total_params = count_modelParams(self.model)
        # Train!
        logger.info('****Start Training!****')
        logger.info(f'Model trainable params:{total_params}')
        logger.info(f'Train example nums:{len(self.train_set)}')
        logger.info(f'Batch size:{self.args.bs}')
        logger.info(f'trainable step:{t_total}')
        logger.info(
            f'gradient accumulate steps:{self.args.grad_accumulate_steps}')
        logger.info(f'logging steps:{self.args.logging_steps}')
        logger.info(f'save steps:{self.args.save_steps}')

        self.model.zero_grad()
        for _ in train_pgb:
            epochs_pgb = tqdm.tqdm(data_iter, desc='iteration')
            for step, batch in enumerate(epochs_pgb):
                self.model.train()
                global_steps += 1
                # load batch_data
                if self.args.batch_first:
                    inputs = tuple(t.to(self.device) for t in batch)
                else:
                    # inputs=[seqlen,bs]
                    inputs = tuple(t.permute(1, 0).to(self.device)
                                   for t in batch[:2])+(batch[-1].to(self.device),)

                # forward pass
                outputs = self.model(*inputs)
                loss = outputs[0]

                # using gradient accumulate
                if self.args.grad_accumulate_steps > 1:
                    loss = loss/self.args.grad_accumulate_steps
                loss.backward()
                total_loss += loss.item()

                # update model
                if global_steps % self.args.grad_accumulate_steps == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.args.max_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()

                # evaluate_model
                if (global_steps % self.args.logging_steps == 0) and (self.args.logging_steps > 0):
                    self.evaluate('eval')

                if (global_steps % self.args.save_steps == 0) and (self.args.save_steps > 0):
                    self.save_model(optimizer, lr_scheduler, global_steps)

                if 0 < t_total < global_steps:
                    epochs_pgb.close()
                    break
            if 0 < t_total < global_steps:
                train_pgb.close()
                break

    def evaluate(self, mode):
        pass

    def save_model(self, optimizer, lr_scheduler, global_steps):
        pass

    @ classmethod
    def reload_data_(cls, model_dir, train_set, eval_set, test_set):
        # check model dir whether exists
        if os.path.exists(model_dir):
            args_path = os.path.join(model_dir, 'train_args.bin')
            args = torch.load(args_path)
            logger.info('****Reload Model success!****')
            return cls(train_set, eval_set, test_set, args,  pretrained_path=args.model_dir)
        else:
            raise FileNotFoundError('Model dir not found!')


class PreTrainedTrainer(BaseTrainer):
    def __init__(self, train_set, eval_set, test_set, args, pretrained_path=None):
        super(PreTrainedTrainer, self).__init__(train_set, eval_set,
                                                test_set, args)

        self.pretrained_path = MODEL_PATH[args.model_type] if pretrained_path is None else pretrained_path
        self.config, _, self.model = MODEL_CLASSES[args.model_type]

        # build Model class
        self.config = self.config.from_pretrained(
            self.pretrained_path, finetuning_task=args.task)
        self.model = self.model.from_pretrained(self.pretrained_path, config=self.config, args=args,
                                                intent_num_labels=len(
                                                    self.intent_vocab),
                                                slot_num_labels=len(self.slot_vocab))
        self.model.to(self.device)

    def evaluate(self, mode):
        if mode == 'eval':
            dataset = self.val_set
        elif mode == 'test':
            dataset = self.test_set
        else:
            raise NameError('Your mode is not exists!')

        sampler = SequentialSampler(dataset)
        data_iter = DataLoader(
            dataset, batch_size=self.args.bs, sampler=sampler)

        intent_preds = None
        intent_label_ids = None
        slot_preds = None
        slot_label_ids = None
        pad_label_id = self.args.ignore_index
        total_loss = 0

        self.model.eval()
        for batch in data_iter:
            batch = tuple(b.to(self.device) for b in batch)
            with torch.no_grad():
                # create inputs
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_type_ids': batch[2],
                    'slot_labels': batch[3],
                    'intent_labels': batch[4], }

                # forward pass
                outputs = self.model(**inputs)
                loss, (intent_logitics, slot_logitics) = outputs[:2]
                total_loss += loss.item()
            # get preds and labels
            # 1.intent preds=[Bs,intent_label_nums]
            intent_labels = inputs['intent_labels'].detach()
            if intent_preds is not None:
                intent_preds = torch.cat(
                    (intent_preds, intent_logitics), dim=0)
                intent_label_ids = torch.cat(
                    (intent_label_ids, intent_labels), dim=0)
            else:
                intent_preds = intent_logitics
                intent_label_ids = intent_labels

            # 2.slot_preds=[Bs,seqLen,slot_num_tags]
            slot_label = inputs['slot_labels'].detach()
            if slot_preds is not None:
                if self.args.use_crf:
                    slot_pred = np.array(
                        self.model.crf_layer.decode(slot_logitics))
                    slot_preds = np.concatenate(
                        (slot_preds, slot_pred), axis=0)
                    slot_label_ids = torch.cat(
                        (slot_label_ids, slot_label), dim=0)
                else:
                    slot_preds = torch.cat((slot_preds, slot_logitics), dim=0)
                    slot_label_ids = torch.cat(
                        (slot_label_ids, slot_label), dim=0)
            else:
                if self.args.use_crf:
                    slot_preds = np.array(
                        self.model.crf_layer.decode(slot_logitics))
                    slot_label_ids = slot_label
                else:
                    slot_preds = slot_logitics
                    slot_label_ids = slot_label

        # compute model metrics
        eval_loss = total_loss/len(data_iter)
        metrics = {'eval_loss': eval_loss, }
        # intent_preds=[Bs,]
        intent_preds = torch.argmax(
            F.softmax(intent_preds, dim=1), dim=1).cpu().numpy()
        intent_label_ids = intent_label_ids.cpu().numpy()
        # slot preds=[Bs,seqLen]
        # filter slot label pad token
        if not self.args.use_crf:
            slot_preds = torch.argmax(
                F.softmax(slot_preds, dim=2), dim=2).cpu().numpy()
        slot_label_ids = slot_label_ids.cpu().numpy()

        slot_label_map = {idx: s for idx, s in enumerate(self.slot_vocab)}
        slot_preds_list = [[] for _ in range(slot_label_ids.shape[0])]
        slot_labels_list = [[] for _ in range(slot_label_ids.shape[0])]

        # convert slot idx to labels
        for i in range(slot_label_ids.shape[0]):
            for j in range(slot_label_ids.shape[1]):
                if slot_label_ids[i, j] != pad_label_id:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i, j]])
                    slot_labels_list[i].append(
                        slot_label_map[slot_label_ids[i, j]])
        # add other Model metrics
        metrics.update(get_modelMetrics(slot_preds_list,
                                        intent_preds, slot_labels_list, intent_label_ids))

        logging.info('**** Start evalulate Model with {mode} data ****')
        logger.info(f'example nums:{len(dataset)}')
        logger.info(f'Batch size:{self.args.bs}')

        for key in sorted(metrics.keys()):
            logger.info(f'{key}={str(metrics[key])}')

        return metrics

    def save_model(self, optimizer, lr_scheduler, global_step):
        # comfirm model dir
        if not os.path.isdir(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        # prepare related Model info.
        optim_params = {
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'global_step': global_step,
        }
        args_savePath = os.path.join(
            self.args.model_dir, 'train_args.bin')
        optim_paramPath = os.path.join(
            self.args.model_dir, 'optim_params.pt')
        # save Model
        try:

            self.model.save_pretrained(self.args.model_dir)
            # save args
            torch.save(self.args, args_savePath)
            # save model info
            torch.save(optim_params, optim_paramPath)
            logger.info(f'Save Model to {self.args.model_dir} Success!')
        except:
            logger.info('Save Model Fail')


class RnnTrainer(BaseTrainer):
    def __init__(self, train_set, eval_set, test_set, args, pretrained_path=None):
        super(RnnTrainer, self).__init__(train_set, eval_set, test_set, args)

        # load Model
        self.config, self.model = MODEL_CLASSES[args.model_type]
        self.config['input_dim'] = len(self.word_vocab)
        self.config['slot_label_nums'] = len(self.slot_vocab)
        self.config['intent_label_nums'] = len(self.intent_vocab)
        if pretrained_path is not None:
            self.model = self.model.reload_model(pretrained_path,args)
        else:
            self.model = self.model(self.config, args)

        self.model.to(self.device)

    def evaluate(self, mode):
        if mode == 'eval':
            dataset = self.val_set
        elif mode == 'test':
            dataset = self.test_set
        else:
            raise NameError('Your mode is not exists!')

        sampler = SequentialSampler(dataset)
        data_iter = DataLoader(dataset, batch_size=self.args.bs,
                               sampler=sampler)

        intent_preds = None
        slot_preds = None
        intent_label_ids = None
        slot_label_ids = None
        pad_label_id = self.args.ignore_index
        total_loss = 0

        self.model.eval()
        for batch in data_iter:
            batch = tuple(b.to(self.device) for b in batch)
            with torch.no_grad():
                # create inputs
                inputs = {
                    'src_tensors': batch[0].permute(1, 0),
                    'trg_tensors': batch[1].permute(1, 0),
                    'intent_labels': batch[2],
                    'teach_ratio': 0.0, }

                # forward pass
                outputs = self.model(**inputs)
                loss, (intent_logitics, slot_logitics) = outputs[:2]
                total_loss += loss.item()

            # get preds and labels
            # 1.intent preds=[Bs,intent_label_nums]
            intent_labels = inputs['intent_labels'].detach()
            if intent_preds is not None:
                intent_preds = torch.cat(
                    (intent_preds, intent_logitics), dim=0)
                intent_label_ids = torch.cat(
                    (intent_label_ids, intent_labels), dim=0)
            else:
                intent_preds = intent_logitics
                intent_label_ids = intent_labels

            # 2.slot_preds=[Bs,seqLen,slot_num_tags]
            #   slot_label=[Bs,seqlen]
            slot_label = inputs['trg_tensors'].permute(1, 0).detach()
            if slot_preds is not None:
                slot_preds = torch.cat((slot_preds, slot_logitics), dim=0)
                slot_label_ids = torch.cat((slot_label_ids, slot_label), dim=0)
            else:
                slot_preds = slot_logitics
                slot_label_ids = slot_label

        # compute model metrics
        eval_loss = total_loss/len(data_iter)
        metrics = {'eval_loss': eval_loss, }

        # intent_preds=[Bs,]
        intent_preds = torch.argmax(
            F.softmax(intent_preds, dim=1), dim=1).cpu().numpy()
        intent_label_ids = intent_label_ids.cpu().numpy()

        # slot preds=[Bs,seqLen]
        # filter slot label pad token
        slot_preds = torch.argmax(
            F.softmax(slot_preds, dim=2), dim=2).cpu().numpy()
        slot_label_ids = slot_label_ids[:, 1:].cpu().numpy()

        slot_label_map = {idx: s for idx, s in enumerate(self.slot_vocab)}
        slot_preds_list = [[] for _ in range(slot_label_ids.shape[0])]
        slot_labels_list = [[] for _ in range(slot_label_ids.shape[0])]

        # convert slot idx to labels
        for i in range(slot_label_ids.shape[0]):
            for j in range(slot_label_ids.shape[1]):
                if slot_label_ids[i, j] != pad_label_id:
                    slot_preds_list[i].append(slot_label_map[slot_preds[i, j]])
                    slot_labels_list[i].append(
                        slot_label_map[slot_label_ids[i, j]])
        # add other Model metrics
        metrics.update(get_modelMetrics(slot_preds_list,
                                        intent_preds, slot_labels_list, intent_label_ids))

        logging.info('**** Start evalulate Model with {mode} data ****')
        logger.info(f'example nums:{len(dataset)}')
        logger.info(f'Batch size:{self.args.bs}')

        for key in sorted(metrics.keys()):
            logger.info(f'{key}={str(metrics[key])}')

        return metrics

    def save_model(self, optimizer, lr_scheduler, global_step):
        # comfirm model dir
        if not os.path.isdir(self.args.model_dir):
            os.makedirs(self.args.model_dir)

        args_savePath = os.path.join(self.args.model_dir, 'train_args.bin')
        model_path = os.path.join(self.args.model_dir, 'pretrain_model.pt')
        config_path = os.path.join(self.args.model_dir, 'config.json')

        # prepare related Model info.
        optim_params = {
            'model_state_dict': self.model.state_dict(),
            'optim_state_dict': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'global_step': global_step,
        }

        # Save data!
        try:
            with open(config_path, 'w', encoding='utf-8') as f_w:
                json.dump(self.config, f_w)

            torch.save(optim_params, model_path)
            torch.save(self.args, args_savePath)
            logger.info(f'Save model to {self.args.model_dir} Success!')
        except:
            logger.info(f'Save model failed!')
