import os
import logging
import copy
import json
import torch
from torch.utils.data import TensorDataset
from utils import get_intent_labels,get_slot_labels
logger=logging.getLogger(__name__)
class InputExamples(object):
    '''
    A sigle string/example for Sentences classification/labeling
    Args:
     guid: Unique Id for example
     words: list.The word of Sequences
     intent_labels:(Optinal) string.intent label of exmaple
     slot_labels:(Optinal) string.Slot label of example
    '''
    def __init__(self,guid,words,intent_labels,slot_labels):
        self.guid=guid
        self.words=words
        self.intent_labels=intent_labels
        self.slot_labels=slot_labels
    def str_to_dict(self):
        '''Serializes instance to python dict'''
        output=copy.deepcopy(self.__dict__)
        return output
    def to_json_string(self):
        return json.dumps(self.str_to_dict,sort_keys=True,indent=2)
    def __repr__(self):
        return str(self.to_json_string())
class Input_features(object):
    def __init__(self,input_tokensIds,attention_mask,token_type_ids,intent_label_ids,slot_label_ids):
        self.input_tokenIds=input_tokensIds
        self.attention_mask=attention_mask
        self.token_type_ids=token_type_ids
        self.intent_label_ids=intent_label_ids
        self.slot_label_ids=slot_label_ids
    def __repr__(self):
        return str(self.str_to_dict)
    def str_to_dict(self):
        '''Serializes instance to python dict'''
        output=copy.deepcopy(self.__dict__)
        return output
    def _to_json_string(self):
        '''Serialize python dict to json string'''
        return json.dumps(self.str_to_dict,sort_keys=True,indent=2)
class JointProcesser(object):
    '''Processor for the JointBERT'''
    def __init__(self,args):
        self.args=args
        self.intent_labels=get_intent_labels(args)
        self.slot_labels=get_slot_labels(args)
        self.input_text_file='seq.in'
        self.slot_labels_file='seq.out'
        self.intent_labels_file='label'
    @classmethod
    def _read_file(cls,file_path,quotechasr=None):
        '''read seperated tab files'''
        with open(file_path,'r',encoding='utf_8') as r_f:
            lines=[]
            for line in r_f:
                lines.append(line.strip())
            return lines
    def create_examples(self,input_texts,intents,slots,set_type):
        '''create examples for train,dev sets'''
        examples=[]
        for i,(inp_t,intents,slots) in enumerate(zip(input_texts,intents,slots)):
            guid='%s-%s'%(set_type,i)
            #1.input word split
            words=inp_t.strip().split()
            #2.intent label
            intent_label=self.intent_labels.index(intents) if intents in self.intent_labels else self.intent_labels.index('UNK')
            #3.slot labels
            slot_labels=[]
            for slot in slots.strip().split():
                slot_labels.append(self.slot_labels.index(slot) if slot in self.slot_labels else self.slot_labels.index('UNK'))
            #require seq len==slot labels len
            assert len(words)==len(slot_labels)
            #append example to examples
            examples.append(InputExamples(guid=guid,words=words,intent_labels=intent_label,slot_labels=slot_labels))
        return examples
    def get_examples(self,mode):
        '''
        Args
         mode:train,dev,test
        '''
        datapath=os.path.join(self.args.data_dir,self.args.task,mode)
        logger.info('LOOking AT{}'.format(datapath))
        return self.create_examples(input_texts=self._read_file(os.path.join(datapath,self.input_text_file)),
                                   intents=self._read_file(os.path.join(datapath,self.intent_labels_file)),
                                   slots=self._read_file(os.path.join(datapath,self.slot_labels_file)),
                                   set_type=mode)
#build data processer
processers={
    'atis':JointProcesser,
    'snips':JointProcesser
}
def convert_example_to_features(examples,
                                max_seqLen,
                                tokenizer,
                                pad_label_id=-100,
                                cls_token_segment_id=0,
                                pad_token_segment_id=0,
                                sequencea_id=0,
                                with_padding_mask=True):
    #setting based for current model type
    cls_token=tokenizer.cls_token
    pad_token=tokenizer.pad_token
    sep_token=tokenizer.sep_token
    unk_token=tokenizer.unk_token
    pad_token_id=tokenizer.pad_token_id
    #整理features
    features=[]
    for ex_id,example in enumerate(examples):
        if ex_id%5000==0:
            logger.info('Writing Example %d of %d'%(ex_id,len(examples)))
        words=example.words
        slots=example.slot_labels
        #Tokenize word by word(NER)
        tokens=[]
        slot_labels=[]
        for word,slot in zip(words,slots):
            subword=tokenizer.tokenize(word)
            if not subword:
                subword=[unk_token]
            tokens.extend(subword)
            #use the real label id for first token,and padding the remain tokens
            slot_labels.extend([int(slot)]+[pad_label_id]*(len(subword)-1))
        #purn special token
        special_token_counts=2
        if len(tokens)>max_seqLen-special_token_counts:
            tokens=tokens[:(max_seqLen-special_token_counts)]
            slot_labels=slot_labels[:(max_seqLen-special_token_counts)]

        #add special token(sep)
        tokens+=[sep_token]
        slot_labels+=[pad_label_id]
        token_type_ids=[sequencea_id]*len(tokens)
        #add cls token
        tokens=[cls_token]+tokens
        slot_labels=[pad_label_id]+slot_labels
        token_type_ids=[cls_token_segment_id]+token_type_ids

        input_tokenIds=tokenizer.convert_tokens_to_ids(tokens)
        #The mask has 1 for real tokens and 0 for pad tokens
        attention_mask=[1 if with_padding_mask else 0]*len(input_tokenIds)
        #padding input seq
        padding_len=max_seqLen-len(input_tokenIds)
        input_tokenIds=input_tokenIds+([pad_token_id]*padding_len)
        token_type_ids=token_type_ids+([pad_token_segment_id]*padding_len)
        attention_mask=attention_mask+([0 if with_padding_mask else 1]*padding_len)
        slot_labels=slot_labels+([pad_label_id]*padding_len)
        
        assert len(input_tokenIds)==max_seqLen, 'Error input_tokenId_len with {} vs {}'.format(len(input_tokenIds),max_seqLen)
        assert len(token_type_ids)==max_seqLen, 'Error token_typeId_len with {} vs {}'.format(len(token_type_ids),max_seqLen)
        assert len(attention_mask)==max_seqLen, 'Error attention_len with {} vs {}'.format(len(attention_mask),max_seqLen)
        assert len(slot_labels)==max_seqLen, 'Error slot_len with {} vs {}'.format(len(slot_labels),max_seqLen)

        intent_label_id=int(example.intent_labels)
        #output example feature in five range
        if ex_id<5:
            logger.info('***Example***')
            logger.info('guId:{}'.format(example.guid))
            logger.info('Input_tokenIds:{}'.format(input_tokenIds))
            logger.info('attention_mask:{}'.format(attention_mask))
            logger.info('input_token_types:{}'.format(token_type_ids))
            logger.info('slot_labelIds:{}'.format(slot_labels))
            logger.info('Intent_ids:{}'.format(intent_label_id))
        features.append(Input_features(input_tokensIds=input_tokenIds,
                                       attention_mask=attention_mask,
                                       token_type_ids=token_type_ids,
                                       intent_label_ids=intent_label_id,
                                       slot_label_ids=slot_labels))
    return features
def load_and_cacheExamples(args,tokenizer,mode):
    #load processor
    processer=processers[args.task](args)
    #load data features for cache file or dataset file
    cache_feature_file=os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            args.task,
            mode,
            list(filter(None,args.model_name_or_path.split('/'))).pop(),
            args.max_seqLen
        )
    )
    if os.path.exists(cache_feature_file):
        logger.info('Loading feature from {}'.format(cache_feature_file))
        features=torch.load(cache_feature_file)
    else:
        #load data features from dataset
        logger.info('Load data feature from dataset in mode:{}'.format(mode))
        if mode=='train':
            examples=processer.get_examples(mode)
        elif mode=='dev':
            examples=processer.get_examples(mode)
        elif mode=='test':
            examples=processer.get_examples(mode)
        else:
            raise Exception('for mode only train,dev,test is available')
        #Use cross entropy ignore index for padding label id so that only real label id can compute loss
        pad_label_id=args.ignore_index
        features=convert_example_to_features(examples=examples,
                                             tokenizer=tokenizer,
                                             max_seqLen=args.max_seqLen,
                                             pad_label_id=pad_label_id)
        logger.info('Save data features into cache file:{}'.format(cache_feature_file))
        torch.save(features,cache_feature_file)
    #convert datafeatures into tensor
    all_input_ids_tensor=torch.tensor([f.input_tokenIds for f in features],dtype=torch.long)
    all_attention_tensor=torch.tensor([f.attention_mask for f in features],dtype=torch.long)
    all_token_type_tensor=torch.tensor([f.token_type_ids for f in features],dtype=torch.long)
    all_slotLabel_tensor=torch.tensor([f.slot_label_ids for f in features],dtype=torch.long)
    all_intentLabel_tensor=torch.tensor([f.intent_label_ids for f in features],dtype=torch.long)
    #build dataset
    dataset=TensorDataset(all_input_ids_tensor,all_attention_tensor,
    all_token_type_tensor,all_intentLabel_tensor,all_slotLabel_tensor)
    return dataset