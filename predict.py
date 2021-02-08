import logging
import os
import argparse
import numpy as np
import torch

from tqdm import tqdm, trange
import torch.nn.functional as F
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from utils import MODEL_CLASSES, get_slot_labels, get_intent_labels, load_tokenizer, init_logger
logger = logging.getLogger(__name__)


def set_device(pred_config):
    return torch.device('cuda:0') if torch.cuda.is_available() and not pred_config.no_cuda else torch.device('cpu')


def read_file(pred_config):
    examples = []
    with open(pred_config.input_file, 'r', encoding='utf-8') as f_r:
        for line in f_r:
            line = line.strip()
            examples.append(line.split())
    return examples


def load_pretrainModel(pred_config, args, intent_labels_num, slot_labels_num):
    # get Model
    config, _, model = MODEL_CLASSES[pred_config.model_type]
    # load model pretrain weight
    config = config.from_pretrained(pred_config.model_dir)
    model = model.from_pretrained(pred_config.model_dir, config, args,
                                  intent_labels_num, slot_labels_num, args.dropout)

    return model


def convert_input_file_to_tensor(input_file, tokenizer, args, pred_config,
                                 pad_label_id=0, pad_token_segment_id=0,
                                 sep_token_segment_id=0, sentA_segment_id=0,
                                 with_pad_mask_zero=True):
    # prepare special token and id
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    # convert all example in input_file
    all_inputIds = []
    all_attnMask = []
    all_segment = []
    all_slotMask = []
    for words in input_file:
        tokens = []
        slot_label_mask = []
        for w in words:
            subwords = tokenizer.tokenize(w)
            if not subwords:
                subwords = [unk_token]
            tokens.extend(subwords)
            slot_label_mask.extend([1]+[0]*(len(subwords)-1))

        # prune subword tokens
        special_token_nums = 2
        max_seqLen = args.max_seqLen
        if len(tokens) > (max_seqLen-special_token_nums):
            tokens = tokens[:(max_seqLen-special_token_nums)]
            slot_label_mask = slot_label_mask[:(max_seqLen-special_token_nums)]

        # add cls,sep token
        tokens += [sep_token]
        slot_label_mask += [pad_label_id]

        tokens = [cls_token]+tokens
        slot_label_mask = [pad_label_id]+slot_label_mask

        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # create sent segment
        segment_ids = [sentA_segment_id]*len(token_ids)
        attn_mask = [1 if with_pad_mask_zero else 0]*len(token_ids)

        # padded token
        pad_len = max_seqLen-len(token_ids)
        tokens += [pad_token_id]*pad_len
        slot_label_mask += [pad_label_id]*pad_len
        segment_ids += [pad_token_segment_id]*pad_len
        attn_mask += [0 if with_pad_mask_zero else 1]*pad_len

        assert len(token_ids) == len(attn_mask) == (
            segment_ids) == (slot_label_mask) == max_seqLen
        all_inputIds.append(token_ids)
        all_attnMask.append(attn_mask)
        all_segment.append(segment_ids)
        all_slotMask.append(slot_label_mask)

    # convert it to tensor
    inputIds_tensors = torch.tensor(all_inputIds, dtype=torch.long)
    attnMask_tensors = torch.tensor(all_attnMask, dtype=torch.long)
    segment_tensors = torch.tensor(all_segment, dtype=torch.long)
    slotMask_tensors = torch.tensor(all_slotMask, dtype=torch.long)

    # create tensor dataset
    dataset = TensorDataset(
        inputIds_tensors, attnMask_tensors, segment_tensors, slotMask_tensors)

    return dataset


def get_predict(model, Dataset, pred_config, args, device):
    # build iterator
    Batch_size = len(Dataset) if len(Dataset) < args.bs else args.bs
    sampler = SequentialSampler(Dataset)
    data_iter = DataLoader(Dataset, batch_size=Batch_size, sampler=sampler)

    slot_preds = None
    intent_preds = None
    model.eval()
    for batch in data_iter:
        # put data in cpu or gpu
        batch = (b.to(device) for b in batch)

        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'slot_labels': None,
            'intent_labels': None,
        }

        outputs = model(**inputs)
        (intent_logitics, slot_logitics) = outputs[1]

        # 1.get intent preds
        # intent_preds=[Bs,intent_num_labels]
        if intent_preds is not None:
            intent_preds = torch.cat((intent_preds, intent_logitics), dim=0)
        else:
            intent_preds = intent_logitics

        # 2.get slot_preds
        # slot_preds=[Bs,seqLen,slot_num_labels]
        slot_labelMask = batch[4] != pred_config.pad_label_id
        attn_mask = batch[1]
        if slot_preds is not None:
            if args.use_crf:
                slot_logitics = model.crf_layer.decode(
                    slot_logitics, mask=attn_mask)
            # filter pad token
            #slot_logitics=[Bs,seqlen] if use_crf else [Bs,seqlen,num_slot_label]
            slot_logitics = slot_logitics[slot_labelMask]
            slot_preds = torch.cat((slot_preds, slot_logitics), dim=0)
        else:
            if args.use_crf:
                slot_logitics = model.crf_layer.decode(
                    slot_logitics, mask=attn_mask)
            # filter pad token
            #slot_logitics=[Bs,seqlen] if use_crf else [Bs,seqlen,num_slot_label]
            slot_logitcs = slot_logitics[slot_labelMask]
            slot_preds = slot_logitics

    # get class index
    intent_preds = torch.argmax(
        F.softmax(intent_preds, dim=1), dim=1).cpu().numpy()
    slot_preds = torch.argmax(
        F.softmax(slot_preds, dim=2), dim=2).cpu().numpy()

    return intent_preds, slot_preds


def convert_to_labels(intent_vocab, slot_vocab, intent_preds, slot_preds):
    assert len(intent_preds) == len(slot_preds), 'pred label nums not equal!'
    intent_labels = []
    slot_labels = []

    for intent, slots in zip(intent_preds, slot_preds):
        slot_sent = []
        for s in slots:
            slot_sent.append(slot_vocab[s])

        intent_labels.append(intent_vocab[int(intent)])
        slot_labels.append(slot_sent)

    return intent_labels, slot_labels


def write_predsFile(pred_config, input_tokens, intent_labels, slot_labels):
    try:
        with open(pred_config.output_file, 'w', encoding='utf-8') as f_w:
            for lines, intent, slots in zip(input_tokens, intent_labels, slot_labels):
                text = ""
                assert len(lines) == len(slots)
                for w, s in zip(lines, slots):
                    if s.startswith('O'):
                        text += w+" "
                    else:
                        text += "[{}:{}] ".format(w, s)
                f_w.write('<{}> -> {} \n'.format(str(intent), text.strip()))
                logger.info(
                    f'Write Model output to {pred_config.output_file} is success!')
    except FileNotFoundError:
        logger.info(f'Output to {pred_config.output_file} is Error!')


def main(pred_config):
    init_logger()
    device = set_device(pred_config)
    # loading train Model args
    args_path = os.path.join(pred_config.model_dir, 'train_args.bin')
    train_args = torch.load(args_path)

    # load tokenizer % vocab
    intent_vocab = get_intent_labels(train_args)
    slot_vocab = get_slot_labels(train_args)
    tokenizer = load_tokenizer(train_args)

    # load preain Model
    model = load_pretrainModel(
        pred_config, train_args, len(intent_vocab), len(slot_vocab))
    model.to(device)

    # read data
    examples = read_file(pred_config)
    # convert data to tensor
    dataset = convert_input_file_to_tensor(
        examples, tokenizer, train_args, pred_config)
    logger.info('***Display PredictInfo***')
    logger.info(f'Predict number:{len(dataset)}')
    logger.info(f'Using Model type:{pred_config.model_type}')
    logger.info(f'Predict max_seqLen:{train_args.max_seqLen}')
    logger.info(f'Whether to use CRF:{train_args.use_crf}')

    # predict!
    logger.info(f'Start to predict using {pred_config.model_type}')
    intent_preds, slot_preds = get_predict(
        model, dataset, pred_config, train_args, device)
    intent_labels, slot_labels = convert_to_labels(
        intent_vocab, slot_vocab, intent_preds, slot_preds)
    # output to file
    write_predsFile(pred_config, examples, intent_labels, slot_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default=None, required=True, help='Input file for prediction')
    parser.add_argument('--output_file', type=str,
                        default=None, required=True, help='output file for prediction')
    parser.add_argument('--model_type', type=str, default=None,
                        required=True, help='selected Model type to load')
    parser.add_argument('--model_dir', type=str, default='./atis_model',
                        required=True, help='load pretrain Model dir')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Control to put data in cpu')
    parser.add_argument('--pad_label_id', default=0, type=int,
                        help="To filter first wordPiece slot tag")
    pred_config = parser.parse_args()
    main(pred_config)
