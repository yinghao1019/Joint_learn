import torch
import torch.nn as nn
from transformers.modeling_bert import BertPreTrainedModel,BertModel
from .module import IntentClassifier,SlotClassifier
from torchcrf import CRF
class JointBert(BertPreTrainedModel):
    def __init__(self,config,num_intent_labels,num_slot_labels,args):
        super(JointBert,self).__init__(config)
        self.args=args
        config.return_dict=True
        self.bert=BertModel(config=config)#load pretrain Model
        self.num_intent_labels=num_intent_labels
        self.num_slot_labels=num_slot_labels
        self.intent_classifier=IntentClassifier(config.hidden_size,self.num_intent_labels,self.args.dropout_rate)
        self.slot_classifier=SlotClassifier(config.hidden_size,self.num_slot_labels,self.args.dropout_rate)
        #determined crf
        if args.use_crf:
            self.crf=CRF(num_tags=num_slot_labels,batch_first=True)
    def forward(self,input_ids,attention_mask,token_type_ids,intent_label_ids,slot_label_ids,return_dict=True):
        return_dict=return_dict if return_dict is not None else self.config.use_return_dict
        output=self.bert(input_ids=input_ids,
                        attention_mask=attention_mask,
                        token_type_ids=token_type_ids,
                        )
        #fetch model last hidden state
        sequence_output=output[0]
        cls_output=output[1]

        intent_logitics=self.intent_classifier(cls_output)
        slots_logitics=self.slot_classifier(sequence_output)
        total_loss=0
        #1.compute intent loss
        if intent_label_ids is not None:
            if self.num_intent_labels==1:
                loss_fct=nn.BCEWithLogitsLoss()
                intent_loss=loss_fct(intent_logitics.view(-1),intent_label_ids.view(-1))
            else:
                loss_fct=nn.CrossEntropyLoss()
                intent_loss=loss_fct(intent_logitics.view(-1,self.num_intent_labels),intent_label_ids.view(-1))
            total_loss+=intent_loss
        #2.compute slot loss
        if slot_label_ids is not None:
            if self.args.use_crf:
                slot_loss=self.crf(slots_logitics,slot_label_ids,mask=attention_mask.byte(),reduction='mean')
                slot_loss=-1*slot_loss#negative log likelihood
            else:
                loss_fct=nn.CrossEntropyLoss()
                if attention_mask is not None:
                    active_loss=attention_mask.view(-1)==1#filter should compute loss position 
                    active_logitics=slots_logitics.view(-1,self.num_slot_labels)[active_loss]#fetch not pad logitics
                    active_slot_labels=slot_label_ids.view(-1)[active_loss]
                    slot_loss=loss_fct(active_logitics,active_slot_labels)
                else:
                    slot_loss=loss_fct(slots_logitics.view(-1,self.num_slot_labels),slot_label_ids.view(-1))
            total_loss+=self.args.slot_loss_coef*slot_loss
        outputs=((intent_logitics,slots_logitics),)+output[2:] #add hidden state & attention
        outputs=(total_loss,)+outputs
        return outputs # return total_loss,logitics,hidden_states,attentions


