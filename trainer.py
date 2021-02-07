import os
import logging
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler
from utils import MODEL_CLASSES, MODEL_PATH, get_intent_labels, get_slot_labels, get_model_metrics
from transformers import AdamW, get_linear_schedule_with_warmup


logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, train_set, eval_set, test_set, args, device):
        self.train_set = train_set
        self.val_set = val_set
        self.test_set = test_set

        self.args = args
        self.intent_vocab = get_intent_labels(args)
        self.slot_vocab = get_slot_labels(args)
        self.device = device

        self.config, _, self.model = MODEL_CLASSES[asgs.model_type]
        self.config = self.config.from_pretrained(MODEL_PATH[args.model_type])
        self.model = self.model.from_pretrained(MODEL_PATH[args.model_type], self.config, args=args,
                                              intent_num_labels=len(
                                                  self.intent_vocab),
                                              slot_num_labels=len(self.slot_vocab), dropout_rate=self.args.dropout)
        self.model.to(self.device)

    def train_model(self):
        # build train iterator
        sampler = RandomSampler(self.train_set)
        data_iter = DataLoader(self.train_set, self.bs, sampler)

        # compute train step
        if self.args.max_step > 0:
            t_toal = self.args.max_step
            self.args.num_train_epochs = t_toal//(
                len(data_iter)//self.grad_accumulate_step)
        else:
            t_total = self.args.num_train_epochs * \
                (len(data_iter)//self.grad_accumulate_step)

        # prepare lr scheduler and optimizer
        no_decay = ['LayerNorm', 'bias']
        param_gropus = [
            {'params': [for n, p in self.model.named_parameters() if not any([nd in n for nd in no_decay])],
             'weight_decay':self.args.weight_decay},
            {'params': [for n, p in self.model.named_parameters() if any([nd in n for nd in no_decay])],
             'weight_decay':0.0},
        ]

        optimizer = AdamW(param_gropus, self.args.train_lr)
        lr_scheduler = get_linear_schedule_with_warmup(
            optimizer, self.args.warm_steps, t_toal)

        # build train_progress
        train_pgb = tqdm.trange(self.args.num_train_epochs, desc='EPOCHS')
        global_steps = 0
        total_loss = 0

        optimizer.zero_grad()
        for _ in train_pgb:
            epochs_pgb = tqdm(data_iter, desc='iteration')
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
                    optimizer.step()
                    lr_scheduler.step()
                    optimizer.zero_grad()
                    global_steps += 1

                # evaluate_model
                if global_steps % self.logging_steps == 0 and self.logging_steps > 0:
                    self.evaluate('eval')

                if global_steps % self.save_steps == 0 and self.save_steps > 0:
                    self.save_model()

                if 0 < self.args.max_step < global_steps:
                    epochs_pgb.close()
                    break
            if 0 < self.args.max_step < global_steps:
                    train_pgb.close()
                    break

    def evaluate(self, mode):
        if mode == 'eval':
            dataset = self.val_set
        elif mode = 'test':
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

            batch = (b.to(device) for b in batch)

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
                if args.use_crf:
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
                if args.use_crf:
                    slot_preds = self.model.crf_layer.decode(
                        slot_logitics, attn_mask)
                    slot_label_ids = slot_label[attn_mask])
                else:
                    slot_preds=slot_logitics[slot_label_mask]
                    slot_label_ids=slot_label[slot_label_mask]


        # compute model metrics
        eval_loss=total_loss/len(data_iter)
        metrics={'eval_loss': eval_loss, }

        # intent_preds=[Bs,]
        intent_preds=torch.argmax(
            F.softmax(intent_preds, dim=1), dim = 1).cpu().numpy()
        intent_label_ids=intent_label_ids.cpu().numpy()
        # slot preds=[Bs,seqLen]
        if not self.args.use_crf:
            slot_preds=torch.argmax(F.softmax(intent_preds, dim=2), dim = 2)

        slot_label_map={idx: s for idx, s in enumerate(self.slot_vocab)}
        slot_preds_list=[[] for _ in range(slot_label_ids.shape[0])]
        slot_labels_list=[[] for _ in range(slot_label_ids.shape[0])]

        # convert slot idx to labels
        for i in range(slot_label_ids.shape[0]):
            for j in range(slot_label_ids.shape[1]):
                slot_preds_list[i].append(slot_label_map[slot_preds[i, j]])
                slot_labels_list[i].append(
                    slot_label_map[slot_label_ids[i, j]])

        # add other Model metrics
        metrics.update(get_model_metrics(slot_preds_list,
                       intent_preds, slot_labels_list, intent_label_ids))

        logging.info('****Model eval metrics****')
        for key in sorted(metrics.keys()):
            logging.info(f'{key}={str(metrics[key])}')

        return metrics
    def save_model(self):



# class Trainer(object):
#     def __init__(self,args,train_set,dev_set,test_set):
#         self.args=args
#         self.train_set=train_set
#         self.dev_set=dev_set
#         self.test_set=test_set
#         #load vocabuary
#         self.intent_vocab_lst=get_intent_labels(args)
#         self.slot_vocab_lst=get_slot_labels(args)
#         #use cross entropy ignore index to only compute real label that contribute to loss
#         self.pad_token_label_id=args.ignore_index
#         #load Model,config,tokenizer
#         self.config,_,self.model_class=MODEL_Classes[args.model_type]
#         self.config=self.config.from_pretrained(args.model_name_or_path,finetuning_task=args.task)
#         self.model_class=self.model_class.from_pretrained(args.model_name_or_path,
#                                               config=self.config,
#                                               num_intent_labels=len(self.intent_vocab_lst),
#                                               num_slot_labels=len(self.slot_vocab_lst),
#                                               args=args)
#         #determined compute model in gpu
#         self.device=torch.device('cuda:0') if torch.cuda.is_available() and args.use_gpu else torch.device('cpu')
#         self.model_class.to(self.device)
#     def train(self):
#         train_sampler=RandomSampler(self.train_set)
#         train_loader=DataLoader(self.train_set,batch_size=self.args.train_batch_size,sampler=train_sampler)
#         if self.args.max_steps>0:
#             t_total=self.args.max_steps
#             self.args.num_train_epochs=self.args.max_steps//(len(train_loader)//self.args.gradient_accumulate_steps)+1
#         else:
#             t_total=(len(train_loader)//self.args.gradient_accumulate_steps)*self.args.num_train_epochs
#         #prepare optimizer and scheduale(linear warm up and decay)
#         no_decay=['bias','LayerNorm.weight']
#         #Layer norm & bias weight decay=0
#         optimizer_grouped_parameters=[
#             {'params':[p for n,p in self.model_class.named_parameters() if not any(nd in n for nd in no_decay)],
#             'weight_decay':self.args.weight_decay},
#             {'params':[p for n,p in self.model_class.named_parameters() if any(nd in n for nd in no_decay)],
#             'weight_decay':0.0}]
#         optimizer=AdamW(optimizer_grouped_parameters,lr=self.args.learning_rate,eps=self.args.adam_epslion)
#         schedular=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=self.args.warmup_steps,num_training_steps=t_total)
#         #Train!
#         logger.info('****Running Training Model****')
#         logger.info('Training Examples size:{}'.format(len(self.train_set)))
#         logger.info('Training epochs:{}'.format(self.args.num_train_epochs))
#         logger.info('Training batch size:{}'.format(self.args.train_batch_size))
#         logger.info('Graient accumulate_steps:{}'.format(self.args.gradient_accumulate_steps))
#         logger.info('Total Optimization steps:{}'.format(t_total))
#         logger.info('Logging steps:{}'.format(self.args.logging_steps))
#         logger.info('save steps:{}'.format(self.args.save_steps))
#         logger.info('model comute in device:{}'.format(self.device))

#         global_step=0
#         tr_loss=0
#         #rerset model params graident to zero
#         self.model_class.zero_grad()
#         train_iterator=trange(int(self.args.num_train_epochs),desc='EPOCHS')
#         for _ in train_iterator:
#             self.model_class.train()#model training Mode
#             epoch_iterator=tqdm(train_loader,desc='iterations')
#             for step,batch in enumerate(epoch_iterator):
#                 batch=tuple(b.to(self.device) for b in batch)
#                 inputs={
#                     'input_ids':batch[0],
#                     'attention_mask':batch[1],
#                     'intent_label_ids':batch[3],
#                     'slot_label_ids':batch[4],
#                     'return_dict':True,
#                 }
#                 if self.args.model_type!='distalbert':
#                     inputs['token_type_ids']=batch[2]
#                 #forward pass
#                 output=self.model_class(**inputs)
#                 loss=output[0]

#                 if self.args.gradient_accumulate_steps>1:
#                     loss=loss/self.args.gradient_accumulate_steps

#                 loss.backward()
#                 tr_loss+=loss.item()
#                 #using gradient clipping to training model stability
#                 if (step+1)%self.args.gradient_accumulate_steps==0:
#                     torch.nn.utils.clip_grad_norm(self.model_class.parameters(),self.args.max_grad_norm)

#                     #update model weight & learning rate
#                     optimizer.step()
#                     schedular.step()
#                     self.model_class.zero_grad()
#                     global_step+=1
#                     if (self.args.logging_steps>0) & (global_step%self.args.logging_steps==0):
#                         print('global_step:',str(global_step))
#                         self.eval('dev')
#                     if (self.args.save_steps>0) & (global_step%self.args.save_steps==0):
#                         self.save_model()
#                 if 0<self.args.max_steps<global_step:
#                     epoch_iterator.close()
#                     break
#             if 0<self.args.max_steps<global_step:
#                 train_iterator.close()
#                 break
#         return global_step,tr_loss/global_step
#     def eval(self,mode):
#         #build data
#         if mode=='test':
#             dataset=self.test_set
#         elif mode=='dev':
#             dataset=self.dev_set
#         eval_sampler=SequentialSampler(dataset)
#         eval_dataloader=DataLoader(dataset,batch_size=self.args.eval_batch_size,sampler=eval_sampler)

#         #eval!
#         logger.info('****Model evaluate on {} dataset'.format(mode))
#         logger.info('Eval examples num:{}'.format(len(dataset)))
#         logger.info('Eval Batch_size num:{}'.format(self.args.eval_batch_size))
#         eval_loss=0
#         eval_steps=0
#         predict_intents=None
#         predict_slots=None
#         out_intent_labels=None
#         out_slot_labels=None

#         self.model_class.eval()
#         for batch in tqdm(eval_dataloader,desc='Evaluation'):
#             with torch.no_grad():
#                 batch=tuple(b.to(self.device) for b in batch)
#                 inputs={
#                     'input_ids':batch[0],
#                     'attention_mask':batch[1],
#                     'intent_label_ids':batch[3],
#                     'slot_label_ids':batch[4],
#                     'return_dict':True,
#                 }
#                 if self.args.model_type!='distalbert':
#                     inputs['token_type_ids']=batch[2]
#                 #forward pass
#                 output=self.model_class(**inputs)
#                 tmp_eval_loss,(intent_logitics,slot_logitics)=output[:2]
#             eval_loss+=tmp_eval_loss.mean().item()
#             eval_steps+=1

#             #1.intent prediction shape=(num_batch_size,intents_label_num)
#             if predict_intents is None:
#                 predict_intents=F.softmax(intent_logitics,dim=1).detach().cpu().numpy()
#                 out_intent_labels=inputs['intent_label_ids'].detach().cpu().numpy()
#             else:
#                 predict_intents=np.append(predict_intents,F.softmax(intent_logitics,dim=1).detach().cpu().numpy(),axis=0)
#                 out_intent_labels=np.append(out_intent_labels,inputs['intent_label_ids'].detach().cpu().numpy(),axis=0)
#             #2.compute slot prediction shape=(num_batch,sentence_len,slot_labe_nums)
#             if predict_slots is None:
#                 if self.args.use_crf:
#                     #decode() in `torchcrf` returns list with best index directly
#                     predict_slots=np.array(self.model_class.crf.decode(slot_logitics))
#                 else:
#                     predict_slots=F.softmax(slot_logitics,dim=2).detach().cpu().numpy()
#                 out_slot_labels=inputs['slot_label_ids'].detach().cpu().numpy()

#             else:

#                 if self.args.use_crf:
#                     #decode() in `torchcrf` returns list with best index directly
#                     predict_slots=np.append(predict_slots,np.array(self.model_class.crf.decode(slot_logitics)),axis=0)
#                 else:
#                     predict_slots=np.append(predict_slots,F.softmax(slot_logitics,dim=2).detach().cpu().numpy(),axis=0)
#                 out_slot_labels=np.append(out_slot_labels,inputs['slot_label_ids'].detach().cpu().numpy(),axis=0)

#         eval_loss=eval_loss/eval_steps
#         results={
#             'eval_loss':eval_loss,
#         }
#         #Intent result
#         predict_intents=np.argmax(predict_intents,axis=1)

#         #Slot preds
#         if not self.args.use_crf:
#             predict_slot=np.argmax(predict_slots,axis=2)
#         slot_label_map={i:label for i,label in enumerate(self.slot_vocab_lst)}
#         out_slot_list=[[] for _ in range(out_slot_labels.shape[0])]
#         slot_predict_list=[[] for _ in range(out_slot_labels.shape[0])]
#         for i in range(out_slot_labels.shape[0]):
#             for j in range(out_slot_labels.shape[1]):
#                 if out_slot_labels[i,j]!=self.pad_token_label_id:
#                     out_slot_list[i].append(slot_label_map[out_slot_labels[i][j]])
#                     slot_predict_list[i].append(slot_label_map[predict_slot[i][j]])

#         #compute eval data precision,recall,f1,semantic frame score
#         total_result=get_model_metrics(predict_intents,slot_predict_list,out_intent_labels,out_slot_list)
#         results.update(total_result)

#         logger.info('****Eval Results****')
#         for key in sorted(results.keys()):
#             logger.info('{}={}'.format(key,results[key]))
#         return results
#     def save_model(self):
#         #save model checkpoint(overwrite)
#         if not os.path.exists(self.args.model_dir):
#             os.makedirs(self.args.model_dir)
#         model_to_save=self.model_class.module if hasattr(self.model_class,'module') else self.model_class
#         model_to_save.save_pretrained(self.args.model_dir)
#         #save args
#         torch.save(self.args,os.path.join(self.args.model_dir,'training_args.bin'))
#         logger.info('save pretrained model to %s'%(self.args.model_dir))
#     def load_model(self):
#         #check whether model_dir exists
#         if os.path.exists(self.args.model_dir):
#             raise Exception('Model not exsits,Please training first')
#         else:
#             try:
#                 self.model_class=self.model_class.from_pretrained(self.args.model_dir,
#                                                                 self.config,
#                                                                 self.intent_vocab_lst,
#                                                                 self.slot_vocab_lst,
#                                                                 self.args)
#                 self.model_class.to(self.device)
#                 logger.info('***Model loading success!****')
#             except:
#                 raise Exception('Some model file be missing')
