import os
import logging
import random
import torch
import numpy as np

from transformers import BertConfig, BertTokenizer
from model import modeling_JointBert, modeling_JointRnn
from seqeval.metrics import f1_score, recall_score, precision_score


def get_word_vocab(args):
    return [w.strip() for w in open(os.path.join(args.data_dir, args.task, args.word_vocab_file))]


MODEL_CLASSES = {
    'joint_bert': (BertConfig, BertTokenizer, modeling_JointBert.JointBert),
    'joint_AttnS2S': (modeling_JointRnn.RnnConfig, modeling_JointRnn.Joint_AttnSeq2Seq),
}
MODEL_PATH = {
    'joint_bert': 'bert-base-uncased',
    'joint_AttnS2S': './atis_model/joint_attnS2S'
}


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]


def load_tokenizer(args):

    return MODEL_CLASSES[args.model_type][1].from_pretrained(MODEL_PATH[args.model_type])


def get_intent_metrics(preds, labels):
    '''get accuracy for sents intent'''
    assert len(preds) == len(labels)
    return {'intent_acc': (preds == labels).mean(), }


def get_slot_metrics(preds, labels):
    '''According IOB1 scheme to evaluate seq'''
    return {
        'slot_f1': f1_score(labels, preds),
        'slot_recall': recall_score(labels, preds),
        'slot_precision': precision_score(labels, preds),
    }


def get_sents_frame_acc(slot_preds, intent_preds, slot_labels, intent_labels):
    # 1.get intent acc
    intent_correct = (intent_preds == intent_labels)
    # 2.compute sents acc
    # confirm same shape
    slot_correct = []
    for slot, label in zip(slot_preds, slot_labels):
        correct = True
        for s, l in zip(slot, label):
            if l != s:
                correct = False
            break
        slot_correct.append(correct)
    slot_correct = np.array(slot_correct)

    # compute semantic
    semantic_acc = np.multiply(intent_correct, slot_correct).mean()

    return {
        'semantic_frame_acc': semantic_acc,
    }


def get_modelMetrics(slot_preds, intent_preds, slot_labels, intent_labels):
    assert len(slot_preds) == len(intent_preds) == len(
        slot_labels) == len(intent_labels)
    metrics = {}
    metrics.update(get_intent_metrics(intent_preds, intent_labels))
    metrics.update(get_slot_metrics(slot_preds, slot_labels))
    metrics.update(get_sents_frame_acc(
        slot_preds, intent_preds, slot_labels, intent_labels))
    return metrics


# set random seed
def set_randomSeed(args):
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.random.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)


# set logger logging message
def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt=r'%m/%d/%Y %H:%M:%S',
                        level=logging.INFO)


# count Model params
def count_modelParams(model):
    total_params = sum([p.numel()
                        for p in model.parameters() if p.requires_grad])
    # downModel_params = sum([sum([p for p in ch.parameters()]) for n, ch in model.named_children()
    #                         if 'bert' not in n])
    return total_params
