import argparse
import logging

from data_loader import load_and_cacheExamples
from utils import load_tokenizer,init_logger,MODEL_Classes,MODEL_PATH,set_seed
from trainer import Trainer

def main(args):
    set_seed(args)
    init_logger()
    #load tokenizer and data
    tokenizer=load_tokenizer(args)
    train_set=load_and_cacheExamples(args,tokenizer,'train')
    dev_set=load_and_cacheExamples(args,tokenizer,'dev')
    test_set=load_and_cacheExamples(args,tokenizer,'test')
    
    model_process=Trainer(args,train_set,dev_set,test_set)
    if args.do_train:
        model_process.train()
    if args.do_eval:
        model_process.load_model()
        model_process.eval('dev')
if __name__=='__main__':
    parser=argparse.ArgumentParser()
    #add model relative infomation argument
    parser.add_argument('--model_type',type=str,default='jointBert',choices=MODEL_PATH.keys(),help='select training model in {}.Default is jointBert'.format(str(MODEL_PATH.keys())))
    parser.add_argument('--max_seqLen',type=int,default=50,help='each example sentence len limited.Default is 50')
    parser.add_argument('--num_train_epochs',type=int,default=10,help='Training model epochs.Only choose it or max_steps to set.Default is 10')
    parser.add_argument('--train_batch_size',type=int,default=32,help='num of examples in one batch.Default is 32')
    parser.add_argument('--eval_batch_size',type=int,default=64,help='num of examples in one batch.Default is 64')
    parser.add_argument('--max_steps',type=int,default=-1,help='If > 0,set max steps to training Model.Only choose it or max_steps to set.Default is -1')
    parser.add_argument('--gradient_accumulate_steps',type=int,default=1,help="Number of updates steps to accumulate before performing a backward/update pass.Default is1")
    parser.add_argument('--warmup_steps',type=int,default=0,help='Linear warm up over warm steps.Default is 0')
    parser.add_argument('--learning_rate',type=float,default=5e-5,help='Optimizer learning rate.Default is 5e-5')
    parser.add_argument('--adam_epslion',type=float,default=1e-8,help='Epsilon for Adam optimizer.Default is 1e-8')
    parser.add_argument('--weight_decay',type=float,default=0.0,help='Weight decay if we apply some.Default is 0')
    parser.add_argument('--max_grad_norm',type=float,default=1.0,help='Max clip Gradient Norm.Default is 1')
    parser.add_argument('--dropout_rate',type=float,default=0.1,help='Dropout for fully connected layer.Default is 0.1')
    parser.add_argument('--save_steps',type=int,default=200,help='Saving model check point in every x training steps.Default is 200')
    parser.add_argument('--logging_steps',type=int,default=200,help='logging model learning information every x training steps.Default is 200')

    #set file attr
    parser.add_argument('--data_dir',type=str,default='./data',help='Saving model check point in every x training steps.Default is ./data')
    parser.add_argument('--model_dir',type=str,required=True,default=None,help='Path to save, load model.Default is None')
    parser.add_argument('--task',type=str,default=None,required=True,choices=['atis','snips'],help='Training Model for task-[atis,snips].Default is None')
    parser.add_argument('--intent_label_files',type=str,default='intent_label.txt',help='intent_labels vocabulary file path.Default is intent_label')
    parser.add_argument('--slot_label_files',type=str,default='slot_label.txt',help='slot_labels vocabulary file path.Default is slot_label')
    parser.add_argument('--seed',type=int,default=1234,help='random seed for initilization')
    #set model perform mode
    parser.add_argument('--do_train',action='store_true',help='Whether to Training Model')
    parser.add_argument('--do_eval',action='store_true',help='Whether to evaluate Model')
    parser.add_argument('--use_gpu',action='store_true',help='Whether is gpu device to Train')
    #crf option
    parser.add_argument('--slot_loss_coef',type=float,default=1.0,help='Coefficient for the slot loss.Default is 1')
    parser.add_argument('--use_crf',action='store_true',help='Whether to use crf')
    #token_id option
    parser.add_argument('--ignore_index',type=int,default=0,help='Specifies a target value that is ignored and does not contribute to the input gradient.Default is 0')
    parser.add_argument('--slot_pad_label',type=str,default='PAD',help='Pad token for slot label pad (to be ignore when calculate loss)')
    
    args=parser.parse_args()

    args.model_name_or_path=MODEL_PATH[args.model_type]
    main(args)