import os
import logging
import copy
import json
import torch
from torch.utils.data import TensorDataset
from utils import get_intent_labels, get_slot_labels
logger = logging.getLogger(__name__)


class Examples:
    def str_to_dict(self):
        '''Serialize example attr to dict'''
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_strng(self):
        return json.dumps(self.str_to_dict, sort_keys=True, indent=2)

    def __repr__(self):
        return str(self.to_json_strng)


class InputExamples(Examples):
    '''
    A single string/example for sentence classification/labeling
    guid:
    words:
    slot_labels:
    intent_label:
    '''

    def __init__(self, guid, words, slot_labels, intent_label):
        self.guid = guid
        self.words = words
        self.slot_labels = slot_labels
        self.intent_label = intent_label


class InputBertFeatures(Examples):
    '''
    A examples feature for sentence classifcaition/tagging

    '''

    def __init__(self, input_tokenIds, attention_mask, token_type_ids, slot_labels, intent_label):
        self.input_tokenIds = input_tokenIds
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.slot_labels = slot_labels
        self.intent_label = intent_label


class InputRnnFeatures(Examples):
    def __init__(self, src_tokenIds, trg_tokenIds, intent_label):
        self.src_tokenIds = src_tokenIds
        self.trg_tokenIds = trg_tokenIds
        self.intent_label = intent_label


class JointProcesser:
    def __init__(self, args):
        self.args = args
        self.intent_vocab = get_intent_labels(args)
        self.slot_vocab = get_slot_labels(args)
        self.input_text_file = 'seq.in'
        self.slot_labels_file = 'seq.out'
        self.intent_label_file = 'label'

    @classmethod
    def _read_file(self, file_path):
        lines = []
        with open(file_path, 'r', encoding='utf-8') as f_r:
            for line in f_r:
                line = line.strip()
                lines.append(line)
        return lines

    def create_examples(self, input_text, intent_labels, slot_labels, set_type):
        examples = []
        for idx, (text, intents, slots) in enumerate(zip(input_text, intent_labels, slot_labels)):
            uid = '%s-%s' % (set_type, idx)
            # split text to tokens
            words = text.strip().split()

            # convert slots to idx
            slot_idxes = []
            for s in slots.strip().split():
                # 1.get slot label index
                if s in self.slot_vocab:
                    slot_id = self.slot_vocab.index(s)
                else:
                    slot_id = self.slot_vocab.index('UNK')
                slot_idxes.append(slot_id)

            # convert intent to idx
            if intents in self.intent_vocab:
                intent_idx = self.intent_vocab.index(intents)
            else:
                intent_idx = self.intent_vocab.index('UNK')

            # assure token nums== slot labels num
            assert len(words) == len(
                slot_idxes), "Input text length don't match slot labels"
            # add to examples
            examples.append(InputExamples(guid=uid, words=words,
                                          slot_labels=slot_idxes, intent_label=intent_idx))
        return examples

    def get_examples(self, mode):
        '''
        Mode:train,dev,test
        '''
        data_path = os.path.join(self.args.data_dir, self.args.task, mode)

        return self.create_examples(
            input_text=self._read_file(os.path.join(
                data_path, self.input_text_file)),
            intent_labels=self._read_file(
                os.path.join(data_path, self.intent_label_file)),
            slot_labels=self._read_file(os.path.join(
                data_path, self.slot_labels_file)),
            set_type=mode)


def convert_to_Seq2SeqFeatures(data, max_seqLen,
                               word_vocab, pad_token_id,
                               src_initTokenId=2, src_endTokenId=3,
                               trg_initTokenId=2, trg_endTokenId=3,
                               ):
    # get special token id
    word_unkId = word_vocab.index('UNK')
    # 整理examples features
    examples = []
    for ex_id, example in enumerate(data):
        if ex_id % 5000 == 0:
            logger.info(f'Already convert {ex_id} examples!')
        tokens = example.words
        slots = example.slot_labels

        # truncate max_seqLen
        special_token_count = 2
        if len(tokens) > (max_seqLen - special_token_count):
            tokens = tokens[:(max_seqLen - special_token_count)]
            slots = slots[:(max_seqLen - special_token_count)]

        # add special tokens
        tokens = ['<sos>'] + tokens + ['<eos>']
        word_tokenIds = [word_vocab.index(
            t) if t in word_vocab else word_unkId for t in tokens]

        slots = [trg_initTokenId] + slots + [trg_endTokenId]

        # pad tokens
        pad_len = max_seqLen - len(tokens)
        word_tokenIds += [pad_token_id] * pad_len
        slots += [pad_token_id] * pad_len
        intent_label = int(example.intent_label)

        assert len(word_tokenIds) == len(slots) == max_seqLen

        if ex_id < 5:
            logger.info('****Display examples****')
            logger.info('Uid:{}'.format(example.guid))
            logger.info('token_ids:{}'.format(word_tokenIds))
            logger.info('slot_labels(trg_tokens):{}'.format(slots))
            logger.info('intent_label:{}'.format(intent_label))

        examples.append(InputRnnFeatures(src_tokenIds=word_tokenIds,
                                         trg_tokenIds=slots,
                                         intent_label=intent_label))

    return examples


def convert_to_BertFeatures(data,
                            max_seqLen,
                            tokenizer,
                            pad_label_id=-100,
                            cls_token_segment_id=0,
                            sep_token_segment_id=0,
                            pad_token_segment_id=0,
                            sentA_id=0,
                            with_padding_mask=True
                            ):
    # get special token
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id
    # 整理example features
    examples = []
    for ex_id, example in enumerate(data):
        if ex_id % 5000 == 0:
            logger.info(f'Already convert {ex_id} examples!')

        # tokenize word to word piece
        tokens = []
        slot_labels = []
        for word, slot in zip(example.words, example.slot_labels):
            subwords = tokenizer.tokenize(word.lower())
            if not subwords:
                subwords = [unk_token]
            tokens.extend(subwords)
            # sure first sub word has slot label and padding remain label
            slot_labels.extend([int(slot)]+[pad_label_id]*(len(subwords)-1))

        assert len(slot_labels) == len(
            tokens), "slot label length not equal word piece"

        # pruncate seqLen
        special_token_counts = 2
        if len(tokens) > (max_seqLen-special_token_counts):
            tokens = tokens[:(max_seqLen-special_token_counts)]
            slot_labels = slot_labels[:(max_seqLen-special_token_counts)]

        # add special token(SEP)
        tokens += [sep_token]
        slot_labels += [pad_label_id]
        token_type_ids = [sentA_id]*len(tokens)

        # add cls token
        tokens = [cls_token]+tokens
        slot_labels = [pad_label_id]+slot_labels
        token_type_ids = [cls_token_segment_id]+token_type_ids

        # convert token to token ids
        token_ids = tokenizer.convert_tokens_to_ids(tokens)
        # create mask attention
        attn_mask = [1 if with_padding_mask else 0]*len(token_ids)

        # padding token
        pad_seqLen = max_seqLen-len(token_ids)
        token_ids += ([pad_token_id]*pad_seqLen)
        slot_labels += ([pad_label_id]*pad_seqLen)
        token_type_ids += ([pad_token_segment_id]*pad_seqLen)
        attn_mask += ([0 if with_padding_mask else 1]*pad_seqLen)

        assert len(token_ids) == max_seqLen, "Error tokenId len with {}vs{}".format(
            len(token_ids), max_seqLen)
        assert len(attn_mask) == max_seqLen, "Error mask_attn len with {}vs{}".format(
            len(attn_mask), max_seqLen)
        assert len(token_type_ids) == max_seqLen, "Error segment len with {}vs{}".format(
            len(token_type_ids), max_seqLen)
        assert len(slot_labels) == max_seqLen, "Error slotLabel len with {}vs{}".format(
            len(slot_labels), max_seqLen)

        intent_label = int(example.intent_label)

        if ex_id < 5:
            logger.info('****Display examples****')
            logger.info('Uid:{}'.format(example.guid))
            logger.info('token_ids:{}'.format(token_ids))
            logger.info('token_type_ids:{}'.format(token_type_ids))
            logger.info('attn mask:{}'.format(attn_mask))
            logger.info('intent_label:{}'.format(intent_label))
            logger.info('slot_labels:{}'.format(slot_labels))

        examples.append(InputBertFeatures(input_tokenIds=token_ids,
                                          attention_mask=attn_mask,
                                          token_type_ids=token_type_ids,
                                          slot_labels=slot_labels,
                                          intent_label=intent_label))

    return examples


# build data proccesser
data_processers = {
    'atis': JointProcesser,
    'snips': JointProcesser,
}


def load_and_cacheExampels(args, tokenizer, mode):
    # load processer
    processer = data_processers[args.task](args)

    # build load and save cached file path
    cached_file_path = os.path.join(args.data_dir, args.task,
                                    'cached_{}_{}_{}_{}.zip'.format(
                                        args.task,
                                        mode,
                                        list(
                                            filter(None, args.model_name_or_path.split('/'))).pop(),
                                        args.max_seqLen))

    if os.path.isfile(cached_file_path):
        # load features
        logger.info('loading data file from {}'.format(cached_file_path))
        features = torch.load(cached_file_path)
    else:
        # create examples
        if mode == 'train':
            examples = processer.get_examples(mode)
        elif mode == 'dev':
            examples = processer.get_examples(mode)
        elif mode == 'test':
            examples = processer.get_examples(mode)
        else:
            raise NameError('Mode args is not train,val,test!')

        # transform example to feature
        pad_label_id = args.ignore_index

        if args.model_type.endswith('bert'):
            features = convert_to_BertFeatures(
                examples, args.max_seqLen, tokenizer, pad_label_id=pad_label_id)
        elif args.model_type.endswith('S2S'):
            features = convert_to_Seq2SeqFeatures(
                examples, args.max_seqLen, tokenizer, pad_token_id=pad_label_id)
        # save to cached file path
        torch.save(features, cached_file_path)
        logger.info(f'Save features to {cached_file_path}')

    # transform features into tensor
    f_tensors = []
    feature_names = vars(features[0]).keys()
    for f_name in feature_names:
        tensors = torch.tensor([getattr(f, f_name)
                                for f in features], dtype=torch.long)
        f_tensors.append(tensors)

    dataset = TensorDataset(*tuple(f_tensors))

    return dataset
