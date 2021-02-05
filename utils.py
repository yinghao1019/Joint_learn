import os
import logging
import random

from transformers import BertConfig,BertTokenizer
from model import modeling_JointBert
MODEL_CLASSES={
    'joint_bert':(BertConfig,BertTokenizer,modeling_JointBert.BertModel)
}
MODEL_PATH={
    'joint_bert':'bert-base-uncased',
}


def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir,args.task,'intent_label.txt','r',encoding='utf-8'))]
def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir,args.task,'slot_label.txt','r',encoding='utf-8'))]

def load_tokenizer(args):
    return MODEL_Classes[args.model_type][1].from_pretrained(MODEL_PATH[args.model_type])


def get_slot_metrics(predict_slots, slot_labels):
    assert len(predict_slots) == len(slot_labels)
    return {
        'slot_precision': precision_score(slot_labels, predict_slots),
        'slot_recall': recall_score(slot_labels, predict_slots),
        'slot_f1_score': recall_score(slot_labels, predict_slots),
    }


def get_intent_metrics(predict_intent, intent_labels):
    acc = np.where(predict_intent == intent_labels, 1, 0).mean()
    return {
        'intent_accuracy': acc,
    }


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def get_sentence_frame_acc(predict_intents, predict_slots, intent_labels, slot_labels):
    '''for this case that  intent and all slots are correct in one sentence'''
    # 1.get the intent comparsion result
    intent_result = predict_intents == intent_labels
    # 2.get the slot comparsion result
    slot_result = []
    for preds, slots in zip(predict_slots, slot_labels):
        assert len(preds) == len(slots)
        one_sentence_correct = True
        for p, l in zip(preds, slots):
            if p != l:
                one_sentence_correct = False
                break
        slot_result.append(one_sentence_correct)
    slot_result = np.array(slot_result)
    semantic_result = np.multiply(intent_result, slot_result)
    return {
        'semantic_acc': semantic_result.mean(),
    }


def get_model_metrics(predict_intent, predict_slots, intent_labels, slot_labels):
    assert len(predict_intent) == len(predict_slots) == len(
        intent_labels) == len(slot_labels)
    result = {}
    # get each metric dict
    intent_result = get_intent_metrics(predict_intent, intent_labels)
    slot_result = get_slot_metrics(predict_slots, slot_labels)
    semantic_result = get_sentence_frame_acc(
        predict_intent, predict_slots, intent_labels, slot_labels)
    # update current metrics
    result.update(intent_result)
    result.update(slot_result)
    result.update(semantic_result)
    return result
# 設定要儲存log的log file


def init_logger():
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt=r'%m/%d/%Y %H:%M:%S', level=logging.INFO)
