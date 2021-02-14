import argparse
import torch

from data_loader import load_and_cacheExampels
from trainer import PreTrainedTrainer, RnnTrainer
from utils import set_randomSeed, init_logger, load_tokenizer, MODEL_PATH, MODEL_CLASSES, get_word_vocab


def main(args):
    set_randomSeed(args)
    init_logger()

    # load tokenizer
    if args.model_type.endswith('S2S'):
        tokenizer = get_word_vocab(args)
    elif args.model_type.endswith('bert'):
        tokenizer = load_tokenizer(args)
    # load dataset
    train_set = load_and_cacheExampels(args, tokenizer, 'train')
    val_set = load_and_cacheExampels(args, tokenizer, 'dev')
    test_set = load_and_cacheExampels(args, tokenizer, 'test')
    # # build train proccess
    if args.model_type.endswith('S2S'):
        proccesser = RnnTrainer(train_set, val_set, test_set, args)
    elif args.model_type.endswith('bert'):
        proccesser = PreTrainedTrainer(train_set, val_set, test_set, args)

    if args.do_train:
        proccesser.train_model()

    if args.do_eval:
        if args.model_type.endswith('S2S'):
            proccesser = RnnTrainer.reload_data_(args.model_dir,
                                                 train_set, val_set, test_set)

        elif args.model_type.endswith('bert'):
            proccesser = PreTrainedTrainer.reload_data_(args.model_dir,
                                                        train_set, val_set, test_set)
        proccesser.evaluate('eval')


if __name__ == '__main__':
    # set run script optinal arg
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='.\data', type=str,
                        help='Root dir path for save data.Default ./data')
    parser.add_argument('--model_dir', default=None, type=str, required=True,
                        help='Path to save training model.Required Argument.')
    parser.add_argument('--task', default=None, required=True, choices=[],
                        type=str, help='Select train Model task:[atis,snips].Required Argument.')
    parser.add_argument('--intent_label_file', default='intent_label.txt',
                        type=str, help='File path for loading intent_label vocab')
    parser.add_argument('--slot_label_file', default='slot_label.txt',
                        type=str, help='File path for loading slot_label vocab ')
    parser.add_argument('--word_vocab_file', default='word_vocab.txt',
                        type=str, help='File path for loading word vocab ')

    parser.add_argument('--model_type', default='joint_bert', type=str, required=True,
                        help='Model type selected in the list:'+','.join(MODEL_CLASSES.keys()))

    parser.add_argument('--random_seed', type=int,
                        default=1234, help='set random seed')
    parser.add_argument('--max_seqLen', type=int, default=50,
                        help='Set max sequence len After tokenize text')

    parser.add_argument('--num_train_epochs', type=int, default=10,
                        help='If>0:set number of train model epochs.')
    parser.add_argument('--max_steps', type=int, default=0,
                        help='If >0:set total_number of train step.Override num_train_epochs')
    parser.add_argument('--warm_steps', type=int,
                        default=0, help='Linear Warm up steps')
    parser.add_argument('--grad_accumulate_steps', type=int,
                        default=1, help='Number of update gradient to accumulate before update model ')
    parser.add_argument('--logging_steps', type=int, default=200,
                        help='Every X train step to logging model info')
    parser.add_argument('--save_steps', type=int, default=200,
                        help='Every X train step to save Model')

    parser.add_argument('--bs', type=int, default=64,
                        help='Train model Batch size')
    parser.add_argument('--train_lr', type=float, default=5e-5,
                        help='Learning rate for AdamW')
    parser.add_argument('--weight_decay', type=float,
                        default=0.0, help='L2 weight regularization for AdamW')
    parser.add_argument('--adam_epsilon', type=float,
                        default=1e-8, help='Epsilon for Adam optimizer.')
    parser.add_argument('--max_norm', type=float,
                        default=1.0, help='Max norm to avoid gradinet exploding')
    parser.add_argument('--dropout', type=float,
                        default=0.1, help='Dropout rate')
    parser.add_argument('--slot_loss_coef', type=float,
                        default=1.0, help='Slot loss coefficient')
    parser.add_argument('--use_crf', action='store_true',
                        help='Whether to using CRF layer for slot pred')

    parser.add_argument('--do_train', action='store_true',
                        help='Whether to run training')
    parser.add_argument('--do_eval', action='store_true',
                        help='Whether to run evaluate')
    parser.add_argument('--no_cuda', action='store_true',
                        help='Control using gpu or cpu to train Model')

    parser.add_argument('--slot_pad_label', type=str, default='PAD',
                        help='Pad token for slot label(Noe contribute loss)')
    parser.add_argument('--ignore_index', type=int, default=0,
                        help='Specifies a target value that not contribute loss and gradient')
    args = parser.parse_args()
    args.model_name_or_path = MODEL_PATH[args.model_type]
    if args.model_type.endswith('S2S'):
        args.batch_first = False
    elif args.model_type.endswith('bert'):
        args.batch_first = True

    main(args)
