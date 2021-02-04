import logging
import os
import argparse
from tqdm import tqdm,trange

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import TensorDataset,DataLoader,SequentialSampler
from utils import init_logger,load_tokenizer,MODEL_Classes,get_intent_labels,get_slot_labels
logger=logging.getLogger(__name__)
def get_device(pred_config):
    return torch.device('cuda:0') if torch.cuda.is_available() and pred_config.use_gpu else torch.device('cpu')
def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir,'training_args.bin'))
def get_model(pred_config,args,device):
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model file don't exists")
    try:
        model=MODEL_Classes[pred_config.model_type][2].from_pretrained(pred_config.model_dir,
                                                            num_intent_labels=len(get_intent_labels(args)),
                                                            num_slot_labels=len(get_slot_labels(args)),
                                                            args=args)
        model.to(device)
        logger.info('Loading model from %s is Success'%(pred_config.model_dir))
    except:
        raise Exception('Some model file will loss')
    return model
def read_inputfile(pred_config):
    lines=[]
    with open(pred_config.input_file,'r',encoding='utf_8') as f_r:
        for line in f_r:
            line=line.strip()
            words=line.split()
            lines.append(words)
    return lines
def convert_inputFile_to_tensorDs(input_file,
                                  pred_config,
                                  tokenizer,
                                  args,
                                  pad_token_label_id,
                                  cls_token_segment_id=0,
                                  pad_token_segment_id=0,
                                  sequenceA_id=0,
                                  with_padding_mask_zero=True):
    #set sepical token type
    pad_token=tokenizer.pad_token
    sep_token=tokenizer.sep_token
    cls_token=tokenizer.cls_token
    unk_token=tokenizer.unk_token
    pad_token_id=tokenizer.pad_token_id
    all_inputIds=[]
    all_attention_maskIds=[]
    all_input_token_typeIds=[]
    all_slot_label_maskIds=[]
    for words in input_file:
        tokens=[]
        slot_label_mask=[]
        for w in words:
            subwords=tokenizer.tokenize(w)
            if not subwords:
                subwords=[unk_token]
            tokens.extend(subwords)
            slot_label_mask.extend([pad_token_label_id+1]+[pad_token_label_id]*(len(subwords)-1))
        #account for speical token
        special_tokens_num=2
        if len(tokens)>args.max_seqLen-special_tokens_num:
            tokens=tokens[:(args.max_seqLen-special_tokens_num)]
            slot_label_mask=slot_label_mask[:(args.max_seqLen-special_tokens_num)]
        #Add special token [SEP]
        tokens+=[sep_token]
        input_token_typeId=[sequenceA_id]*len(tokens)
        slot_label_mask+=[pad_token_label_id]
        #Add special token [CLS]
        tokens=[cls_token]+tokens
        input_token_typeId=[cls_token_segment_id]+input_token_typeId
        slot_label_mask=[pad_token_label_id]+slot_label_mask
        
        input_ids=tokenizer.convert_tokens_to_ids(tokens)

        #The mask has 1 for real token and 0 for pad token.Only real token is attened
        attention_maskId=[1 if with_padding_mask_zero else 0]*len(input_ids)
        #paddding tokens to same seq len
        padding_len=args.max_seqLen-len(input_ids)
        input_ids=input_ids+padding_len*[pad_token_id]
        input_token_typeId=input_token_typeId+padding_len*[pad_token_segment_id]
        slot_label_mask=slot_label_mask+padding_len*[pad_token_label_id]
        attention_maskId=attention_maskId+padding_len*[0 if with_padding_mask_zero else 1]

        all_inputIds.append(input_ids)
        all_attention_maskIds.append(attention_maskId)
        all_input_token_typeIds.append(input_token_typeId)
        all_slot_label_maskIds.append(slot_label_mask)

    #convert data to tensordataset
    all_input_tensors=torch.tensor(all_inputIds,dtype=torch.long)
    all_attention_mask_tensors=torch.tensor(all_attention_maskIds,dtype=torch.long)
    all_input_token_type_tensors=torch.tensor(all_input_token_typeIds,dtype=torch.long)
    all_slot_label_mask_tensors=torch.tensor(all_slot_label_maskIds,dtype=torch.long)

    dataset=TensorDataset(all_input_tensors,all_attention_mask_tensors,
    all_input_token_type_tensors,all_slot_label_mask_tensors)

    return dataset
def predict(pred_config):
    #get parameters
    args=get_args(pred_config)
    device=get_device(pred_config)
    model=get_model(pred_config,args,device)
    logger.info(args)
    #get related data info
    pad_token_label_id=args.ignore_index
    tokenizer=load_tokenizer(args)
    inputfile=read_inputfile(pred_config)
    intent_label_lst=get_intent_labels(args)
    slot_label_lst=get_slot_labels(args)

    dataset=convert_inputFile_to_tensorDs(inputfile,pred_config,tokenizer,args,pad_token_label_id=pad_token_label_id)
    #predict 
    sampler=SequentialSampler(dataset)
    data_loader=DataLoader(dataset,batch_size=pred_config.batch_size,sampler=sampler)

    all_slot_label_mask=None
    intent_preds=None
    slot_preds=None

    model.eval()
    with torch.no_grad():
        for batch in tqdm(data_loader,desc='Predictions'):
            #batch data contain 1.input_ids 2.attention_mask 3.segment_ids 4.slot_label_mask
            batch=tuple(b.to(device) for b in batch)
            inputs={
                'input_ids':batch[0],
                'attention_mask':batch[1],
                'intent_label_ids':None,
                'slot_label_ids':None,
                'return_dict':True
                }
            #determine whether model is distBert
            if args.model_type !='distalbert':
                inputs['token_type_ids']=batch[2]
            #Model return output contain 1.total_loss 2.logitics 3.hidden_states 4.attention weight
            outputs=model(**inputs)

            _,(intent_logitics,slot_logitics)=outputs[:2]

            if intent_preds is None:
                intent_preds=F.softmax(intent_logitics,dim=1).detach().cpu().numpy()
            else:
                intent_preds=np.append(intent_preds,F.softmax(intent_logitics,dim=1).detach().cpu().numpy(),axis=0)
            if slot_preds is None:
                #decodes in torchcrf will return  list with best index directly
                if args.use_crf:
                    slot_preds=np.array(model.crf.decode(slot_logitics))
                else:
                    slot_preds=F.softmax(slot_logitics,dim=2).detach().cpu().numpy()
                all_slot_label_mask=batch[3].cpu().numpy()
            else:
                #decodes in torchcrf will return  list with best index directly
                if args.use_crf:
                    slot_preds=np.append(slot_preds,np.array(model.crf.decode(slot_logitics)),axis=0)
                else:
                    slot_preds=np.append(slot_preds,F.softmax(slot_logitics,dim=2).detach().cpu().numpy(),axis=0)
                all_slot_label_mask=np.append(all_slot_label_mask,batch[3].cpu().numpy(),axis=0)
            
    intent_preds=np.argmax(intent_preds,axis=1)
    if not args.use_crf:
        slot_preds=np.argmax(slot_preds,axis=2)

    slot_label_map={i:s for i,s in enumerate(slot_label_lst)}
    slot_preds_lst=[[] for _ in range(all_slot_label_mask.shape[0])]
    print(all_slot_label_mask.shape)
    #maping Non mask slot id to string
    for i in range(all_slot_label_mask.shape[0]):
        for j in range(all_slot_label_mask.shape[1]):
            if all_slot_label_mask[i,j]!=pad_token_label_id:
                slot_preds_lst[i].append(slot_label_map[slot_preds[i][j]])

    #write pred intent label & slots in sentences to file(txt)
    logger.info('Write preds(intent,slots) to %s'%(pred_config.output_file))
    with open(pred_config.output_file,'w',encoding='utf_8') as f_w:
        for lines,intent,slots in zip(inputfile,intent_preds,slot_preds_lst):
            text=""
            #add word with corresponding to slot
            assert len(lines)==len(slots),'Word length:{} vs slot length:{}'.format(len(lines),len(slots))
            for word,slot in zip(lines,slots):
                if slot=='O':
                    text+=word+" "
                else:
                    text+="[{}:{}]".format(word,slot)
            f_w.write('<{}>-->{} \n'.format(intent_label_lst[int(intent)],text.strip()))

    logger.info('Write pred to file Done!')

if __name__=='__main__':
    init_logger()
    parser=argparse.ArgumentParser()
    parser.add_argument('--input_file',default='sample_pred_in.txt',type=str,help='Input sample for prediction')
    parser.add_argument('--output_file',default='sample_pred_out.txt',type=str,help='Output sample for prediction')
    parser.add_argument('--model_dir',default='./atis_model',type=str,help='Path to save ,load model')
    parser.add_argument('--model_type',type=str,help='load model type')
    parser.add_argument('--batch_size',default=256,type=int,help='Batch_size for prediction')
    parser.add_argument('--use_gpu',action='store_true',help='Determined to use gpu for Prediction')

    pred_config=parser.parse_args()
    predict(pred_config)

                

