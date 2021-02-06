import os
import logging
import random
import numpy as np

from transformers import BertConfig, BertTokenizer
from model import modeling_JointBert
from seqeval.metrics import f1_score, recall_score, precision_score
MODEL_CLASSES = {
    'joint_bert': (BertConfig, BertTokenizer, modeling_JointBert.JointBert)
}
MODEL_PATH = {
    'joint_bert': 'bert-base-uncased',
}


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, 'intent_label.txt', 'r', encoding='utf-8'))]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, 'slot_label.txt', 'r', encoding='utf-8'))]


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
    intent_correct = intent_preds == intent_labels
    # 2.compute sents acc
    # confirm same shape
    assert slot_preds.shape[0] == slot_labels.shape[0]
    assert slot_preds.shape[1] == slot_labels.shape[1]

    slot_correct = np.all(slot_preds == slot_labels, axis=1)
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
