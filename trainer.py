import os
import logging
import numpy as np
import random
import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from utils import MODEL_CLASSES, MODEL_PATH, get_intent_labels, get_slot_labels, get_modelMetrics, count_modelParams
from transformers import AdamW, get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, train_set, eval_set, test_set, args, device, pretrained_path=None):
        self.train_set = train_set
        self.val_set = eval_set
        self.test_set = test_set

        self.args = args
        self.intent_vocab = get_intent_labels(args)
        self.slot_vocab = get_slot_labels(args)
        self.device = device

        self.pretrain_path = pretrained_path if pretrained_path is not None else MODEL_PATH[
            args.model_type]
        self.config, _, self.model = MODEL_CLASSES[args.model_type]
        self.config = self.config.from_pretrained(self.pretrain_path)
        self.model = self.model.from_pretrained(self.pretrain_path, self.config, args=args,
                                                intent_num_labels=len(
                                                    self.intent_vocab),
                                                slot_num_labels=len(self.slot_vocab), dropout_rate=self.args.dropout)
        self.model.to(self.device)

    def train_model(self):
        # build train iterator
        sampler = RandomSampler(self.train_set)
        data_iter = DataLoader(self.train_set, self.args.bs, sampler)

        # compute train step
        if self.args.max_step > 0:
            t_toal = self.args.max_step
            self.args.num_train_epochs = t_toal//(
                len(data_iter)//self.args.grad_accumulate_step)
        else:
            t_total = self.args.num_train_epochs * \
                (len(data_iter)//self.args.grad_accumulate_step)

        # prepare lr scheduler and optimizer
        no_decay = ['LayerNorm', 'bias']
        param_gropus = [
            {'params': [p for n, p in self.model.named_parameters() if not any([nd in n for nd in no_decay])],
             'weight_decay':self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any([nd in n for nd in no_decay])],
             'weight_decay':0.0},
        ]

        optimizer = AdamW(param_gropus, self.args.train_lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, self.args.warm_steps, t_toal)

        # build train_progress
        train_pgb = tqdm.trange(self.args.num_train_epochs, desc='EPOCHS')
        global_steps = 0
        total_loss = 0

        # count params
        total_params, dowm_params = count_modelParams(self.model)
        # Train!
        logger.info('****Start Training!****')
        logger.info(f'Model trainable params:{total_params}')
        logger.info(f'Train example nums:{len(self.train_set)}')
        logger.info(f'Batch size:{self.args.bs}')
        logger.info(f'trainable step:{t_total}')
        logger.info(
            f'Gradient accunulate step:{self.args.grad_accumulate_step}')
        logger.info(f'logging step:{self.args.logging_steps}')
        logger.info(f'save step:{self.args.save_steps}')

        self.model.zero_grad()
        for _ in train_pgb:
            epochs_pgb = tqdm.tqdm(data_iter, desc='iteration')
            for step, batch in enumerate(epochs_pgb):
                self.model.train()

                # load batch_data
                batch = (t.to(self.device)
                         for t in batch)  # put tensor to gpu or cpu

                # create inputs
                inputs = {
                    'input_ids': batch[0],
                    'attention_mask': batch[1],
                    'token_typ_ids': batch[2],
                    'slot_labels': batch[3],
                    'intent_labels': batch[4]
                }
                # forward pass
                outputs = self.model(**inputs)
                loss = outputs[0]

                # using gradient accumulate
                if self.args.grad_accumulate_step > 1:
                    loss = loss/self.args.grad_accumulate_step
                loss.backward()
                total_loss += loss.item()

                # update model
                if global_steps % self.args.grad_accumulate_step == 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model_parameters(), self.max_norm)
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_steps += 1

                # evaluate_model
                if global_steps % self.args.logging_steps == 0 and self.args.logging_steps > 0:
                    self.evaluate('eval')

                if global_steps % self.args.save_steps == 0 and self.args.save_steps > 0:
                    self.save_model(optimizer, lr_scheduler, global_steps)

                if 0 < self.args.max_step < global_steps:
                    epochs_pgb.close()
                    break
            if 0 < self.args.max_step < global_steps:
                train_pgb.close()
                break

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
        pad_label_id = self.args.pad_label_id
        total_loss = 0
        self.model.eval()
        for batch in data_iter:

            batch = (b.to(self.device) for b in batch)

            # create inputs
            inputs = {
                'input_ids': batch[0],
                'attention_mask': batch[1],
                'token_typ_ids': batch[2],
                'slot_labels': batch[3],
                'intent_labels': batch[4],
            }

            # forward pass
            outputs = self.model(**inputs)

            loss, (intent_logitics, slot_logitics) = outputs[:2]
            total_loss += loss.item()

            # get preds and labels
            # 1.intent preds=[Bs,intent_label_nums]
            intent_label = inputs['intent_labels'].detach()
            intent_label_mask = intent_label != pad_label_id
            if intent_preds is not None:
                intent_preds = torch.cat(
                    (intent_preds, intent_logitics[intent_label_mask]), dim=0)
                intent_label_ids = torch.cat(
                    intent_label_ids, intent_label[intent_label_mask], dim=0)
            else:
                intent_preds = intent_logitics[intent_label_mask]
                intent_label_ids = intent_label_ids[intent_label_mask]

            # 2.slot_preds=[Bs,seqLen,slot_num_tags]
            slot_label = inputs['slot_labels'].detach()
            slot_label_mask = slot_label != pad_label_id
            attn_mask = inputs['attention_mask']
            if slot_preds is not None:
                if self.args.use_crf:
                    slot_pred = self.model.crf_layer.decode(
                        slot_logitics, attn_mask)
                    slot_preds = torch.cat(
                        (slot_preds, slot_pred[slot_label_mask]), dim=0)
                    slot_label_ids = torch.cat(
                        (slot_label_ids, slot_label[attn_mask]), dim=0)
                else:
                    slot_preds = torch.cat(
                        (slot_preds, slot_logitics[slot_label_mask]), dim=0)
                    slot_label_ids = torch.cat(
                        (slot_label_ids, slot_label[slot_label_mask]), dim=0)
            else:
                if self.args.use_crf:
                    slot_preds = self.model.crf_layer.decode(
                        slot_logitics, attn_mask)[slot_label_mask]
                    slot_label_ids = slot_label[slot_label_mask]
                else:
                    slot_preds = slot_logitics[slot_label_mask]
                    slot_label_ids = slot_label[slot_label_mask]

        # compute model metrics
        eval_loss = total_loss/len(data_iter)
        metrics = {'eval_loss': eval_loss, }

        # intent_preds=[Bs,]
        intent_preds = torch.argmax(
            F.softmax(intent_preds, dim=1), dim=1).cpu().numpy()
        intent_label_ids = intent_label_ids.cpu().numpy()
        # slot preds=[Bs,seqLen]
        if not self.args.use_crf:
            slot_preds = torch.argmax(F.softmax(intent_preds, dim=2), dim=2)

        slot_label_map = {idx: s for idx, s in enumerate(self.slot_vocab)}
        slot_preds_list = [[] for _ in range(slot_label_ids.shape[0])]
        slot_labels_list = [[] for _ in range(slot_label_ids.shape[0])]

        # convert slot idx to labels
        for i in range(slot_label_ids.shape[0]):
            for j in range(slot_label_ids.shape[1]):
                slot_preds_list[i].append(slot_label_map[slot_preds[i, j]])
                slot_labels_list[i].append(
                    slot_label_map[slot_label_ids[i, j]])

        # add other Model metrics
        metrics.update(get_modelMetrics(slot_preds_list,
                                        intent_preds, slot_labels_list, intent_label_ids))

        logging.info('****Model eval metrics****')
        for key in sorted(metrics.keys()):
            logger.info(f'{key}={str(metrics[key])}')

        return metrics

    def save_model(self, optimizer, lr_scheduler, global_step):
        # comfirm model dir
        if not os.path.isdir(self.args.save_model_dir):
            os.mkdir(self.args.save_model_dir)
        # prepare related Model info.
        optim_params = {
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'global_step': global_step,
        }
        args_savePath = os.path.join(
            self.args.save_model_dir, 'train_args.bin')
        optim_paramPath = os.path.join(
            self.args.save_model_dir, 'optim_params.pt')
        # save Model
        try:

            self.model.save_pretrained(self.args.model_dir)
            # save args
            torch.save(self.args, args_savePath)
            # save model info
            torch.save(optim_params, optim_paramPath)
            logger.info(f'Save Model to {self.args.save_model_dir} Success!')
        except:
            logger.info('Save Model Fail')

    @ classmethod
    def reload_model_(cls, model_dir, train_set, eval_set, test_set, device):
        # check model dir whether exists
        if os.path.exists(model_dir):
            try:
                args_path = os.path.join(model_dir, 'train_args.bin')
                args = torch.load(args_path)
                logger.info('****Reload Model success!****')
                return cls(train_set, eval_set, test_set, args, device, pretrained_path=args.save_model_dir)
            except:
                logger.info('Model some file was missed!')
        else:
            raise FileNotFoundError('Model dir not found!')
