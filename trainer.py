import os
import logging
from tqdm import tqdm,trange#loading training process bar

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import DataLoader,SequentialSampler,RandomSampler
from utils import get_intent_labels,get_slot_labels,MODEL_Classes,MODEL_PATH,get_model_metrics
from transformers import AdamW,get_linear_schedule_with_warmup,BertConfig
logger=logging.getLogger(__name__)

class Trainer(object):
    def __init__(self,args,train_set,dev_set,test_set):
        self.args=args
        self.train_set=train_set
        self.dev_set=dev_set
        self.test_set=test_set
        #load vocabuary
        self.intent_vocab_lst=get_intent_labels(args)
        self.slot_vocab_lst=get_slot_labels(args)
        #use cross entropy ignore index to only compute real label that contribute to loss
        self.pad_token_label_id=args.ignore_index
        #load Model,config,tokenizer
        self.config,_,self.model_class=MODEL_Classes[args.model_type]
        self.config=self.config.from_pretrained(args.model_name_or_path,finetuning_task=args.task)
        self.model_class=self.model_class.from_pretrained(args.model_name_or_path,
                                              config=self.config,
                                              num_intent_labels=len(self.intent_vocab_lst),
                                              num_slot_labels=len(self.slot_vocab_lst),
                                              args=args)
        #determined compute model in gpu
        self.device=torch.device('cuda:0') if torch.cuda.is_available() and args.use_gpu else torch.device('cpu')
        self.model_class.to(self.device)
    def train(self):
        train_sampler=RandomSampler(self.train_set)
        train_loader=DataLoader(self.train_set,batch_size=self.args.train_batch_size,sampler=train_sampler)
        if self.args.max_steps>0:
            t_total=self.args.max_steps
            self.args.num_train_epochs=self.args.max_steps//(len(train_loader)//self.args.gradient_accumulate_steps)+1
        else:
            t_total=(len(train_loader)//self.args.gradient_accumulate_steps)*self.args.num_train_epochs
        #prepare optimizer and scheduale(linear warm up and decay)
        no_decay=['bias','LayerNorm.weight']
        #Layer norm & bias weight decay=0
        optimizer_grouped_parameters=[
            {'params':[p for n,p in self.model_class.named_parameters() if not any(nd in n for nd in no_decay)],
            'weight_decay':self.args.weight_decay},
            {'params':[p for n,p in self.model_class.named_parameters() if any(nd in n for nd in no_decay)],
            'weight_decay':0.0}]
        optimizer=AdamW(optimizer_grouped_parameters,lr=self.args.learning_rate,eps=self.args.adam_epslion)
        schedular=get_linear_schedule_with_warmup(optimizer,num_warmup_steps=self.args.warmup_steps,num_training_steps=t_total)
        #Train!
        logger.info('****Running Training Model****')
        logger.info('Training Examples size:{}'.format(len(self.train_set)))
        logger.info('Training epochs:{}'.format(self.args.num_train_epochs))
        logger.info('Training batch size:{}'.format(self.args.train_batch_size))
        logger.info('Graient accumulate_steps:{}'.format(self.args.gradient_accumulate_steps))
        logger.info('Total Optimization steps:{}'.format(t_total))
        logger.info('Logging steps:{}'.format(self.args.logging_steps))
        logger.info('save steps:{}'.format(self.args.save_steps))
        logger.info('model comute in device:{}'.format(self.device))

        global_step=0
        tr_loss=0
        #rerset model params graident to zero
        self.model_class.zero_grad()
        train_iterator=trange(int(self.args.num_train_epochs),desc='EPOCHS')
        for _ in train_iterator:
            self.model_class.train()#model training Mode
            epoch_iterator=tqdm(train_loader,desc='iterations')
            for step,batch in enumerate(epoch_iterator):
                batch=tuple(b.to(self.device) for b in batch)
                inputs={
                    'input_ids':batch[0],
                    'attention_mask':batch[1],
                    'intent_label_ids':batch[3],
                    'slot_label_ids':batch[4],
                    'return_dict':True,
                }
                if self.args.model_type!='distalbert':
                    inputs['token_type_ids']=batch[2]
                #forward pass
                output=self.model_class(**inputs)
                loss=output[0]

                if self.args.gradient_accumulate_steps>1:
                    loss=loss/self.args.gradient_accumulate_steps

                loss.backward()
                tr_loss+=loss.item()
                #using gradient clipping to training model stability
                if (step+1)%self.args.gradient_accumulate_steps==0:
                    torch.nn.utils.clip_grad_norm(self.model_class.parameters(),self.args.max_grad_norm)
                    
                    #update model weight & learning rate
                    optimizer.step()
                    schedular.step()
                    self.model_class.zero_grad()
                    global_step+=1
                    if (self.args.logging_steps>0) & (global_step%self.args.logging_steps==0):
                        print('global_step:',str(global_step))
                        self.eval('dev')
                    if (self.args.save_steps>0) & (global_step%self.args.save_steps==0):
                        self.save_model()
                if 0<self.args.max_steps<global_step:
                    epoch_iterator.close()
                    break
            if 0<self.args.max_steps<global_step:
                train_iterator.close()
                break
        return global_step,tr_loss/global_step
    def eval(self,mode):
        #build data
        if mode=='test':
            dataset=self.test_set
        elif mode=='dev':
            dataset=self.dev_set
        eval_sampler=SequentialSampler(dataset)
        eval_dataloader=DataLoader(dataset,batch_size=self.args.eval_batch_size,sampler=eval_sampler)

        #eval!
        logger.info('****Model evaluate on {} dataset'.format(mode))
        logger.info('Eval examples num:{}'.format(len(dataset)))
        logger.info('Eval Batch_size num:{}'.format(self.args.eval_batch_size))
        eval_loss=0
        eval_steps=0
        predict_intents=None
        predict_slots=None
        out_intent_labels=None
        out_slot_labels=None

        self.model_class.eval()
        for batch in tqdm(eval_dataloader,desc='Evaluation'):
            with torch.no_grad():
                batch=tuple(b.to(self.device) for b in batch)
                inputs={
                    'input_ids':batch[0],
                    'attention_mask':batch[1],
                    'intent_label_ids':batch[3],
                    'slot_label_ids':batch[4],
                    'return_dict':True,
                }
                if self.args.model_type!='distalbert':
                    inputs['token_type_ids']=batch[2]
                #forward pass
                output=self.model_class(**inputs)
                tmp_eval_loss,(intent_logitics,slot_logitics)=output[:2]
            eval_loss+=tmp_eval_loss.mean().item()
            eval_steps+=1

            #1.intent prediction shape=(num_batch_size,intents_label_num)
            if predict_intents is None:
                predict_intents=F.softmax(intent_logitics,dim=1).detach().cpu().numpy()
                out_intent_labels=inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                predict_intents=np.append(predict_intents,F.softmax(intent_logitics,dim=1).detach().cpu().numpy(),axis=0)
                out_intent_labels=np.append(out_intent_labels,inputs['intent_label_ids'].detach().cpu().numpy(),axis=0)
            #2.compute slot prediction shape=(num_batch,sentence_len,slot_labe_nums)
            if predict_slots is None:
                if self.args.use_crf:
                    #decode() in `torchcrf` returns list with best index directly
                    predict_slots=np.array(self.model_class.crf.decode(slot_logitics))
                else:
                    predict_slots=F.softmax(slot_logitics,dim=2).detach().cpu().numpy()
                out_slot_labels=inputs['slot_label_ids'].detach().cpu().numpy()
            
            else:
                
                if self.args.use_crf:
                    #decode() in `torchcrf` returns list with best index directly
                    predict_slots=np.append(predict_slots,np.array(self.model_class.crf.decode(slot_logitics)),axis=0)
                else:
                    predict_slots=np.append(predict_slots,F.softmax(slot_logitics,dim=2).detach().cpu().numpy(),axis=0)
                out_slot_labels=np.append(out_slot_labels,inputs['slot_label_ids'].detach().cpu().numpy(),axis=0)
            
        eval_loss=eval_loss/eval_steps
        results={
            'eval_loss':eval_loss,
        }
        #Intent result
        predict_intents=np.argmax(predict_intents,axis=1)

        #Slot preds
        if not self.args.use_crf:
            predict_slot=np.argmax(predict_slots,axis=2)
        slot_label_map={i:label for i,label in enumerate(self.slot_vocab_lst)}
        out_slot_list=[[] for _ in range(out_slot_labels.shape[0])]
        slot_predict_list=[[] for _ in range(out_slot_labels.shape[0])]
        for i in range(out_slot_labels.shape[0]):
            for j in range(out_slot_labels.shape[1]):
                if out_slot_labels[i,j]!=self.pad_token_label_id:
                    out_slot_list[i].append(slot_label_map[out_slot_labels[i][j]])
                    slot_predict_list[i].append(slot_label_map[predict_slot[i][j]])

        #compute eval data precision,recall,f1,semantic frame score
        total_result=get_model_metrics(predict_intents,slot_predict_list,out_intent_labels,out_slot_list)
        results.update(total_result)

        logger.info('****Eval Results****')
        for key in sorted(results.keys()):
            logger.info('{}={}'.format(key,results[key]))
        return results
    def save_model(self):
        #save model checkpoint(overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save=self.model_class.module if hasattr(self.model_class,'module') else self.model_class
        model_to_save.save_pretrained(self.args.model_dir)
        #save args
        torch.save(self.args,os.path.join(self.args.model_dir,'training_args.bin'))
        logger.info('save pretrained model to %s'%(self.args.model_dir))
    def load_model(self):
        #check whether model_dir exists
        if os.path.exists(self.args.model_dir):
            raise Exception('Model not exsits,Please training first')
        else:
            try:
                self.model_class=self.model_class.from_pretrained(self.args.model_dir,
                                                                self.config,
                                                                self.intent_vocab_lst,
                                                                self.slot_vocab_lst,
                                                                self.args)
                self.model_class.to(self.device)
                logger.info('***Model loading success!****')
            except:
                raise Exception('Some model file be missing')