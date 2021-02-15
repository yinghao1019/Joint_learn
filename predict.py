import logging
import os
import argparse
import numpy as np
import torch

from tqdm import tqdm, trange
import torch.nn.functional as F
from torch.utils.data import TensorDataset, SequentialSampler, DataLoader
from utils import MODEL_CLASSES, get_slot_labels, get_intent_labels, load_tokenizer, init_logger, get_word_vocab
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


def load_pretrainModel(pred_config, args, intent_num_labels, slot_num_labels):
    # get Model
    config, _, model = MODEL_CLASSES[pred_config.model_type]
    # load model pretrain weight
    if pred_config.model_type.endswith('bert'):
        config = config.from_pretrained(pred_config.model_dir)
        model = model.from_pretrained(pred_config.model_dir, config=config, args=args,
                                      intent_num_labels=intent_num_labels, slot_num_labels=slot_num_labels)
    else:
        model = model.reload_model(pred_config.model_dir, args)

    return model


def convert_input_file_to_RnnTensor(input_file, word_vocab, args, pred_config, pad_token_id):
    # get special token Id
    sos_token_id = word_vocab.index('<sos>')
    eos_token_id = word_vocab.index('<eos>')
    pad_token_id = pad_token_id

    all_inputIds = []
    all_inputLens = []
    for words in input_file:

        special_counts = 2
        # truncate max seqLen
        max_seqLen = args.max_seqLen
        if len(words) > (max_seqLen-special_counts):
            words = words[:(max_seqLen-special_counts)]

        # add special token
        words = ['<sos>']+words+['<eos>']
        # convert to wordId
        wordIds = [word_vocab.index(
            w) if w in word_vocab else word_vocab.index('UNK') for w in words]
        # compute origin len
        origin_len = len(wordIds)

        # pad seqLen
        pad_len = max_seqLen-origin_len
        wordIds += [pad_token_id]*pad_len

        assert len(wordIds) == max_seqLen, 'token Id len:{} vs max_len:{}'

        all_inputIds.append(wordIds)
        all_inputLens.append(origin_len-1)  # get decoder output max len

    # convert it to tensor
    inputIds_tensors = torch.tensor(all_inputIds, dtype=torch.long)
    inputLens_tensors = torch.tensor(all_inputLens, dtype=torch.long)
    # create tensor dataset
    dataset = TensorDataset(inputIds_tensors, inputLens_tensors)

    return dataset


def convert_input_file_to_BertTensor(input_file, tokenizer, args, pred_config,
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
        token_ids += [pad_token_id]*pad_len
        slot_label_mask += [pad_label_id]*pad_len
        segment_ids += [pad_token_segment_id]*pad_len
        attn_mask += [0 if with_pad_mask_zero else 1]*pad_len

        assert len(token_ids) == len(attn_mask) == len(
            segment_ids) == len(slot_label_mask) == max_seqLen
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


def get_s2s_predict(model, Dataset, pred_config, args, slot_vocab, device):
    slot_allPreds = []
    slot_allMask = []
    intent_allPreds = None
    sampler = SequentialSampler(Dataset)
    data_loader = DataLoader(Dataset, batch_size=1, sampler=sampler)

    model.eval()
    for data in data_loader:
        inputs = {
            'src_tensors': data[0].permute(1, 0).to(device),
            'origin_len': data[1].item(),
            'trg_initTokenId': slot_vocab.index('<sos>'),
            'trg_endTokenId': slot_vocab.index('<eos>'), }
        with torch.no_grad():
            intent_logitcis, slot_pred = model.get_predict(**inputs)

        assert len(slot_pred) == (
            inputs['origin_len']), f'predictLen:{len(slot_pred)} vs origin len:{data[1].item()}'

        # 1.get intent
        if intent_allPreds is not None:
            intent_allPreds = torch.cat((intent_allPreds, intent_logitcis))
        else:
            intent_allPreds = intent_logitcis

        # 2.get slot_preds amd slot_mask
        slot_mask = [1]*len(slot_pred[:-1])

        slot_allPreds.append(slot_pred[:-1])
        slot_allMask.append(slot_mask)

    # get max prob intent class
    intent_allPreds = torch.argmax(torch.softmax(
        intent_allPreds, dim=1), dim=1).cpu().numpy()

    return intent_allPreds, slot_allPreds, slot_allMask


def get_pretrain_predict(model, Dataset, pred_config, args, device):
    # build iterator
    Batch_size = len(Dataset) if len(
        Dataset) < pred_config.bs else pred_config.bs
    sampler = SequentialSampler(Dataset)
    data_iter = DataLoader(Dataset, batch_size=Batch_size, sampler=sampler)

    slot_preds = None
    intent_preds = None
    model.eval()
    for batch in data_iter:
        # put data in cpu or gpu
        batch = tuple(b.to(device) for b in batch)
        inputs = {
            'input_ids': batch[0],
            'attention_mask': batch[1],
            'token_type_ids': batch[2],
            'slot_labels': None,
            'intent_labels': None,
        }
        with torch.no_grad():
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
        if slot_preds is not None:
            # add slot preds
            if args.use_crf:
                slot_pred = np.array(model.crf_layer.decode(
                    slot_logitics))
                slot_preds = np.concatenate((slot_preds, slot_pred), axis=0)
            else:
                slot_preds = torch.cat((slot_preds, slot_logitics), dim=0)

            # add slot_mask label
            slot_maskLabels = torch.cat((slot_maskLabels, batch[3]), dim=0)
        else:
            # add slot preds
            if args.use_crf:
                slot_preds = np.array(model.crf_layer.decode(
                    slot_logitics))
            else:
                slot_preds = slot_logitics

            # add slot_mask label
            slot_maskLabels = batch[3]

    # get class index
    intent_preds = torch.argmax(
        F.softmax(intent_preds, dim=1), dim=1).cpu().numpy()
    if not args.use_crf:
        slot_preds = torch.argmax(
            F.softmax(slot_preds, dim=2), dim=2).cpu().numpy()
    slot_maskLabels = slot_maskLabels.cpu().numpy()

    return intent_preds, slot_preds, slot_maskLabels


def convert_to_labels(intent_vocab, slot_vocab, intent_preds,
                      slot_preds, slot_masks, pad_label_id):
    assert len(intent_preds) == len(slot_preds), 'pred label nums not equal!'
    intent_labels = []
    slot_labels = []

    for intent, slots, mask in zip(intent_preds, slot_preds, slot_masks):
        slot_sent = []
        for s, m in zip(slots, mask):
            if m != pad_label_id:
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

    # load labels
    intent_vocab = get_intent_labels(train_args)
    slot_vocab = get_slot_labels(train_args)

    # load preain Model
    model = load_pretrainModel(
        pred_config, train_args, len(intent_vocab), len(slot_vocab))
    model.to(device)

    # read data
    examples = read_file(pred_config)
    pad_label_id = pred_config.pad_label_id

    logger.info(f'Start to predict using {pred_config.model_type}')
    if pred_config.model_type.endswith('S2S'):
        # convert data to tensor
        tokenizer = get_word_vocab(train_args)
        dataset = convert_input_file_to_RnnTensor(examples, tokenizer, train_args,
                                                  pred_config, pad_token_id=pad_label_id)

        # get predict!
        intent_preds, slot_preds, slot_masks = get_s2s_predict(model, dataset, pred_config,
                                                               train_args, slot_vocab, device)
    elif pred_config.model_type.endswith('bert'):
        # convert data to tensor
        tokenizer = load_tokenizer(train_args)
        dataset = convert_input_file_to_BertTensor(examples, tokenizer, train_args,
                                                   pred_config, pad_label_id=pad_label_id)

        # get predict!
        intent_preds, slot_preds, slot_masks = get_pretrain_predict(model, dataset, pred_config,
                                                                    train_args, device)

    logger.info('***Display PredictInfo***')
    logger.info(f'Predict number:{len(dataset)}')
    logger.info(f'Predict max_seqLen:{train_args.max_seqLen}')
    logger.info(f'Whether to use CRF:{train_args.use_crf}')

    intent_labels, slot_labels = convert_to_labels(intent_vocab, slot_vocab,
                                                   intent_preds, slot_preds,
                                                   slot_masks, pad_label_id)
    # output to file
    write_predsFile(pred_config, examples, intent_labels, slot_labels)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str,
                        default=None, required=True, help='Input file for prediction.Required argument')
    parser.add_argument('--output_file', type=str,
                        default=None, required=True, help='output file for prediction.Required argument')
    parser.add_argument('--model_type', type=str, default=None, choices=MODEL_CLASSES.keys(),
                        required=True, help='Model type selected in list:'+','.join(MODEL_CLASSES.keys()))
    parser.add_argument('--model_dir', type=str, default=None,
                        required=True, help='load pretrain Model dir.Required argument')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Control to put data in cpu')
    parser.add_argument('--pad_label_id', default=0, type=int,
                        help="To filter first wordPiece slot tag")
    parser.add_argument('--bs', default=16, type=int,
                        help="Batch size for predict")
    pred_config = parser.parse_args()
    main(pred_config)
