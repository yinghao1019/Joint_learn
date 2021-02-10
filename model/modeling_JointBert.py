import torch
import torch.nn as nn
from torch.nn import functional as F
from transformers import BertModel, BertPreTrainedModel
from torchcrf import CRF
from .module import intent_classifier, slot_classifier


class JointBert(BertPreTrainedModel):
    def __init__(self, config, args, intent_num_labels, slot_num_labels):
        super(JointBert, self).__init__(config)
        self.intent_num_labels = intent_num_labels
        self.slot_num_labels = slot_num_labels
        self.args = args
        self.config = config

        # create layer
        self.bert = BertModel(config=config)
        self.intent_classifier = intent_classifier(self.config.hidden_size,
                                                   intent_num_labels, self.args.dropout)
        self.slot_classifier = slot_classifier(self.config.hidden_size,
                                               slot_num_labels, self.args.dropout)
        # build conditional random field
        if args.use_crf:
            self.crf_layer = CRF(slot_num_labels, batch_first=True)

    def forward(self, input_ids, attention_mask, token_type_ids, slot_labels, intent_labels):
        # get bert hidden state
        outputs = self.bert(
            input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)

        hidden_output = outputs[0]  # sequence of hiddn state
        pooled_output = outputs[1]  # cls token hidden state

        # intent_logitics=[Bs,intent_num_labels]
        # slot_logitics=[Bs,max_seqLen,slot_num_labels]
        slot_logitics = self.slot_classifier(hidden_output)
        intent_logticis = self.intent_classifier(pooled_output)

        total_loss = 0
        # compute
        if intent_labels is not None:
            if self.intent_num_labels == 1:
                intent_loss = F.mse_loss(intent_logticis, intent_labels)
            else:
                intent_loss = F.cross_entropy(intent_logticis, intent_labels)
            total_loss += intent_loss

        if slot_labels is not None:
            criterion = nn.CrossEntropyLoss(
                ignore_index=self.args.ignore_index)
            if self.args.use_crf:
                slot_loss = self.crf_layer(slot_logitics, slot_labels,
                                           mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1*slot_loss  # negative log likelihood
            else:
                if attention_mask is not None:
                    active_mask = attention_mask.view(-1) == 1
                    active_logictics = slot_logitics.view(
                        -1, self.slot_num_labels)[active_mask]
                    active_labels = slot_labels.view(-1)[active_mask]
                    slot_loss = criterion(
                        active_logictics, active_labels)
                else:
                    slot_loss = criterion(slot_logitics.view(-1, self.slot_num_labels),
                                          slot_labels.view(-1))
            total_loss += self.args.slot_loss_coef*slot_loss

        outputs = ((intent_logticis, slot_logitics),)+outputs[2:]
        outputs = (total_loss,)+outputs

        # return total_loss,logitics,hidden_state,attention, #logticis contain intent_logitics,slot_logiticis
        return outputs
