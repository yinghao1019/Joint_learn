import unicodedata
import string
import re
import random
from io import open
import logging
import argparse
from tqdm import tqdm
import os
import numpy as np

import torch
from torch import nn, optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from utils import init_logger
logger = logging.getLogger(__name__)

# 過濾指定長度的sentences 以及sentences prefix
eng_prefixes = (
    "i am ", "i m ",
    "he is", "he s ",
    "she is", "she s ",
    "you are", "you re ",
    "we are", "we re ",
    "they are", "they re "
)


class Lang_vocab:
    def __init__(self, name):
        self.name = name
        self.word2index = {'SOS': 0, 'EOS': 1}
        self.index2word = {0: 'SOS', 1: 'EOS'}
        self.wordCount = {}
        self.n_words = 2

    def addSent(self, sentences):
        for word in sentences.split(' '):
            self.addWord(word)

    def addWord(self, word):
        # calculate word index and count
        if word not in self.word2index:
            self.index2word[self.n_words] = word
            self.word2index[word] = self.n_words
            self.wordCount[word] = 1
            self.n_words += 1
        else:
            self.wordCount[word] += 1
# create Encoder


class Encoder_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(Encoder_RNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)

    def forward(self, input, last_hidden):
        input_embed = self.embed(input).view(1, 1, -1)
        # output shape (batch,time_stamp,hidden)

        output, hidden_state = self.gru(input_embed, last_hidden)
        return output, hidden_state

    def init_hidden(self, device):
        return torch.zeros(1, 1, self.hidden_dim, device=device)
# Create decoder


class Decoder_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(Decoder_RNN, self).__init__()
        self.args = args
        self.hidden_dim = hidden_dim
        self.embed = nn.Embedding(input_dim, hidden_dim)
        self.input_transform = nn.Linear(hidden_dim*2, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim)
        self.linear = nn.Linear(hidden_dim, output_dim)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, prev_hidden, encoder_hiddens, context_vector):
        embed_input = self.embed(inputs).view(1, 1, -1)
        # concat context vector and transform
        embed_input = self.input_transform(
            torch.cat((embed_input, context_vector), dim=2))
        output, hidden_state = self.gru(embed_input, prev_hidden)
        output_logitics = self.linear(output[0])
        # return softmax output(computed with log),logisitcs,hidden_state
        return (self.logsoftmax(output_logitics), output_logitics, hidden_state)

    def init_hidden(self, device):
        return torch.zeros(1, 1, self.hidden_dim, device=device)
# create attnDecoder


class AttnDecoder_RNN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, args):
        super(AttnDecoder_RNN, self).__init__()
        self.args = args
        self.max_seqLen = args.max_seqLen
        self.dropout_p = args.droprate_p
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim

        self.embed = nn.Embedding(input_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, 1)
        self.attn_layer = nn.Linear(2*hidden_dim, args.max_seqLen)
        self.input_attnCombine = nn.Linear(2*hidden_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(self.dropout_p)
        # activation
        self.softmax = nn.Softmax(dim=2)
        self.relu = nn.ReLU(2)
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, prev_hidden, encoder_hiddens):
        embed_input = self.embed(inputs).view(1, 1, -1)

        embed_input = self.dropout(embed_input)
        # compute attention weight shape=(seqlen,bs,hid_dim)
        attn_weight = F.softmax(self.attn_layer(
            torch.cat((embed_input, prev_hidden), dim=2)), dim=2)
        # compute attention repr=attenion_weight(timstep,bs,encoder_hid_num)*encoderHidden (encoder_hiddnum,encoder_hidden dim)
        attn_repr = torch.mm(attn_weight[0], encoder_hiddens)
        # combine attention & input_embed into input_vector
        input_vector = self.input_attnCombine(
            torch.cat((embed_input, attn_repr.unsqueeze(dim=0)), dim=2))
        input_vector = self.relu(input_vector)

        output_logitics, hidden_state = self.gru(input_vector, prev_hidden)
        output_logitics = self.output_layer(output_logitics[0])
        return (self.logsoftmax(output_logitics), output_logitics, hidden_state, attn_weight)

    def init_hidden(self, device):
        return torch.zeros(1, 1, self.hidden_dim, device=device)


def UnicodetoAscii(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')


def NormalizeString(s):
    s = UnicodetoAscii(s.lower().strip())
    s = re.sub(r'[.!?]', ' \1', s)
    s = re.sub(r'[^a-zA-Z.!?]+', r' ', s)
    return s


def _read_file_to_pair(lang1, lang2, Reversed=True):
    # read file like object full content
    lines = open('./data/%s-%s.txt' % (lang1, lang2), mode='r',
                 encoding='utf_8').read().strip().split('\n')
    sent_pairs = [[NormalizeString(sents)
                   for sents in line.split('\t')]for line in lines]
    if Reversed:
        sents_pair = [pair.reverse() for pair in sent_pairs]
        Input_lang = Lang_vocab(lang2)
        Output_lang = Lang_vocab(lang1)
    else:
        sents_pair = [pair for pair in sent_pairs]
        Input_lang = Lang_vocab(lang1)
        Output_lang = Lang_vocab(lang2)
    return Input_lang, Output_lang, sent_pairs


def filter_pair(p):
    return len(p[0].split(' ')) < args.max_seqLen and \
        len(p[1].split(' ')) < args.max_seqLen and \
        p[1].startswith(eng_prefixes)


def filter_pairs(pairs):
    return [pair for pair in pairs if filter_pair(pair)]


def data_prepare(lang1, lang2, Reversed=None):
    '''
    Args
     lang1:str.Input language for translate
     lang2:str.Target language is translated
     Reversed:Boolean.while True,reversed translate order(lang2-lang1) vice versa.Default is None
    Return
     Lang1_vocab,Lang2_vocab:vocab object
     lang_pairs:list of list.which index 0 is lang1 sentences,lang2 sents
    '''
    # read data
    Lang1_vocab, Lang2_vocab, lang_pairs = _read_file_to_pair(
        lang1, lang2, Reversed=Reversed)

    logger.info(
        'Read {}-{} pairs sentences data num:{}'.format(lang1, lang2, len(lang_pairs)))

    # filter <max length pair
    lang_pairs = filter_pairs(lang_pairs)
    # build each lang vocab
    for pair in lang_pairs:
        Lang1_vocab.addSent(pair[0])
        Lang2_vocab.addSent(pair[1])
    logger.info(f'{Lang1_vocab.name} vocab num:{Lang1_vocab.n_words}')
    logger.info(f'{Lang2_vocab.name} vocab num:{Lang2_vocab.n_words}')
    return Lang1_vocab, Lang2_vocab, lang_pairs
# convert word 2 index


def ConvertSentIndex(pair, lang_vocab):
    return [lang_vocab.word2index[w] for w in pair]


def tensorFromSents(pair, device, lang_vocab):
    # convert to index
    sent_index = ConvertSentIndex(pair.split(' '), lang_vocab)
    # add Eos token
    sent_index.append(EOS_token)
    return torch.tensor(sent_index, dtype=torch.long, device=device)
# each sequence pair convert to tensor


def tensorFrompair(pair, lang1_vocab, lang2_vocab):
    '''
    Convert string sentences to corresponding indexes.and each sentences end will add eos token_id(1)
    Args
     pair:list of list .which index 0 is first langauge seq,1 is second seq
     lang1_vocab,lang2_vocab:vocabulary object.
    Return
     tensor pair
    '''
    input_tensors = tensorFromSents(pair[0], device, lang1_vocab).view(-1, 1)
    output_tensors = tensorFromSents(pair[1], device, lang2_vocab).view(-1, 1)
    return input_tensors, output_tensors


def training_model(encoder, decoder, input_tensor,
                   target_tensor, encoder_optimizer,
                   decoder_optimizer, criterion,
                   args, schedule_samplingProb):
    # init hidden state
    hidden_state = encoder.init_hidden(device)
    input_length = input_tensor.size(0)
    target_length = target_tensor.size(0)
    # storage each time step hidden state
    encoder_outputs = torch.zeros(
        args.max_seqLen, encoder.hidden_dim, device=device)
    loss = 0
    # encoder stage
    for t in range(input_length):
        output, hidden_state = encoder(input_tensor[t], hidden_state)
        encoder_outputs[t] = output[0, 0]
    decoder_input = torch.tensor(
        [[SOS_token]], device=device, dtype=torch.long)

    decoder_hidden = hidden_state
    for ti in range(target_length):
        output = decoder(decoder_input, decoder_hidden,
                         encoder_outputs, hidden_state)
        decoder_output, decoder_logitics, decoder_hidden = output[:3]
        # select next input token() for decoder which depend on schedule sampling prob

        if random.random() < schedule_samplingProb:
            # use teacher forcing
            decoder_input = target_tensor[ti]
        else:
            decoder_input = torch.argmax(
                nn.functional.softmax(decoder_logitics, dim=1), dim=1)
        # compute decoder & target loss
        loss += criterion(decoder_output, target_tensor[ti])
    loss.backward()
    encoder_optimizer.step()  # update encoder wieght
    decoder_optimizer.step()  # update decoder weight

    # clear parameters gradient
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    return loss.item()/target_length


def display_paramsNum(model):
    total_params = 0
    for sub_name, sub_model in model.named_children():
        model_params = sum([param.nelement()
                            for param in sub_model.parameters()])  # count parameters
        logger.info(f'sub model name:{sub_name}-->num_params:{model_params}')
        total_params += model_params
    logger.info(f'Model total params num:{total_params}')


def get_modelPredict(encoder, decoder, Input_tensor, args):
    input_seqLen = Input_tensor.size(0)
    hidden_state = encoder.init_hidden(device)
    seq_predictions = []
    seq_attn = None
    # storage each time step hidden state
    encoder_outputs = torch.zeros(
        args.max_seqLen, encoder.hidden_dim, device=device)

    encoder.eval()
    decoder.eval()
    with torch.no_grad():
        logger.info('Input sequence Encode stage....')
        for t in range(input_seqLen):
            output_hidden, hidden_state = encoder(
                Input_tensor[t], hidden_state)
            # append every time token repr.
            encoder_outputs[t] = output_hidden[0, 0]
        logger.info(f'Input seq each time hidden_state:{encoder_outputs}')

        decode_hidden = hidden_state  # storage encoder last hidden
        decoder_input = torch.tensor(
            [[args.SOS_tokenId]], dtype=torch.long, device=device)

        for i in range(args.max_seqLen):
            output = decoder(decoder_input, decode_hidden,
                             encoder_outputs, hidden_state)
            decoder_output, decoder_logitics, decoder_hidden = output[:4]
            # output shape =(timestamp,batch_size)
            decoder_input = torch.argmax(
                nn.functional.softmax(decoder_logitics, dim=1), dim=1)
            # add each time attention
            # if seq_attn is not None:
            #     seq_attn = torch.cat(
            #         (seq_attn, torch.squeeze(attn_output, dim=0)), 0)
            # else:
            #     seq_attn = torch.squeeze(attn_output, dim=0)
            # add seq predict
            seq_predictions.append(decoder_input.detach().cpu().item())
            if decoder_input.item() == args.EOS_tokenId:
                break
    return seq_predictions, seq_attn


def save_model(encoder, decoder, encode_optim, decoder_optim, iters, save_dir):
    # 判斷save dir是否存在
    if os.path.isdir(save_dir):
        logger.info('model dir is existed')
    else:
        logger.info('build Model sace dir')
        os.mkdir(save_dir)
    model_info = {
        'Encoder': encoder.state_dict(),
        'Decoder': decoder.state_dict(),
        'encoder_optimizer': encode_optim.state_dict(),
        'decoder_optimizer': decoder_optim.state_dict(),
        'iters': iters
    }
    torch.save(model_info, os.path.join(save_dir, 'attn_model.pt'))
    print('Save Model success!')


def load_model_state(model_dir, model, model_config, model_name, model_type):
    model = model(**model_config)
    # load dict from model_dir
    if os.path.exists(model_dir):
        try:
            state_dict = torch.load(os.path.join(
                model_dir, model_type), map_location=device)
            logger.info(
                f'model load from {os.path.join(model_dir,model_type)}')
            model.load_state_dict(state_dict[model_name])  # load model
            return model, state_dict
        except:
            raise Exception(f'Model load Error')
    else:
        raise Exception(f'No Model dir exists')


def trainIters(encoder, decoder, data_pairs, lang1_vocab, lang2_vocab,
               display_lossIter, model_config, args, is_saved=True):
    # load model
    encoder = encoder(**model_config['Encoder'])
    decoder = decoder(**model_config['Decoder'])
    encoder.to(device)
    decoder.to(device)
    # count model parameters
    logger.info('******Encoder parameters num******')
    display_paramsNum(encoder)
    logger.info('******decoder parameters num******')
    display_paramsNum(decoder)

    # set model parameters
    encoder_optim = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optim = optim.Adam(decoder.parameters(), lr=args.learning_rate)
    criterion = nn.NLLLoss()
    plot_loss = []
    per_iterloss = 0
    iter_pairs = [random.choice(data_pairs) for _ in range(args.max_iters)]
    iter_progress = tqdm(iter_pairs, desc='Training_iters')

    # training
    for i, pair in enumerate(iter_progress):
        encoder.train()
        decoder.train()
        # get Input,output tensor
        input_tensor, target_tensor = tensorFrompair(
            pair, lang1_vocab, lang2_vocab)

        # display data info
        if i < 5:
            logger.info(
                f'Input data:{input_tensor} shape:{input_tensor.size()}')
            logger.info(
                f'Output data:{target_tensor} shape:{target_tensor.size()}')
        # set schedule sampling prob decay
        sample_prob = float(
            args.init_prob*float((args.max_iters-(i+1))/args.max_iters))
        loss = training_model(encoder, decoder, input_tensor, target_tensor,
                              encoder_optim, decoder_optim, criterion, args, sample_prob)
        per_iterloss += loss

        # compute loss at every display iters
        if i % display_lossIter-1 == 0:
            iter_loss = per_iterloss/(i+1)
            plot_loss.append(iter_loss)
            seq_preds, seq_attn = get_modelPredict(
                encoder, decoder, input_tensor, args)
            print('Input seq:', pair[0])
            print('True seq:', pair[1])
            print('Predict seq:', ''.join(
                [lang2_vocab.index2word[w_index] for w_index in seq_preds]))
            logger.info(f'iters:{i+1}-total_loss:{iter_loss} cur_loss:{loss}')
            per_iterloss = 0

        # save model ckpoint
        if i % args.save_iters-1 == 0 and is_saved:
            save_model(encoder, decoder, encoder_optim,
                       decoder_optim, i+1, args.model_dir)


def test_stage(encoder, decoder, lang_pairs, lang1_vocab, lang2_vocab, model_dir, n_iters=10):
    # load model init params & config
    model_config = torch.load(os.path.join(
        model_dir, 'model_training_args.bin'))
    encoder, _ = load_model_state(
        model_dir, encoder, model_config['Encoder'], 'Encoder', 'model.pt')
    decoder, _ = load_model_state(
        model_dir, decoder, model_config['Decoder'], 'Decoder', 'model.pt')
    encoder.to(device)
    decoder.to(device)
    # radom choice N_iters seq to predict
    iter_pairs = [random.choice(lang_pairs) for _ in range(n_iters)]

    for i, pairs in enumerate(iter_pairs):
        Input_tensor = tensorFromSents(pairs[0], device, lang1_vocab)
        logger.info(
            f'time-{i} Translate Seq Len:{len(pairs[0])} \n Seq:{pairs[0]}\n Seq index:{Input_tensor}')

        # model predict
        predict_seq, predict_attn = get_modelPredict(
            encoder, decoder, Input_tensor, args)
        # display predict
        logger.info(f"time-{i} Target seq:{pairs[1]}\n \
        Predict seq:{' '.join([lang2_vocab.index2word[token_id] for token_id in predict_seq])}\n \
        Predict seq index:{predict_seq}")


def main(args):
    # read data
    Lang1_vocab, Lang2_vocab, lang_pairs = data_prepare(
        args.input_lang, args.trans_lang, Reversed=True)
    logger.info(
        f'random pairs state:{random.choice(lang_pairs)} pairs total num:{len(lang_pairs)}')
    model_config = {
        'Encoder': {'input_dim': Lang1_vocab.n_words, 'hidden_dim': 256},
        'Decoder': {'input_dim': Lang2_vocab.n_words, 'hidden_dim': 256, 'output_dim': Lang2_vocab.n_words, 'args': args}
    }
    torch.save(model_config, r'.\data\training_args.bin')
    # training model step
    if args.do_train:
        trainIters(Encoder_RNN, Decoder_RNN, lang_pairs,
                   Lang1_vocab, Lang2_vocab, 5000, model_config, args)
    if args.do_eval:
        test_stage(Encoder_RNN, Decoder_RNN, lang_pairs,
                   Lang1_vocab, Lang2_vocab, args.model_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', default=1e-2,
                        type=float, help='model Learning rate.Defalut is 0.01')
    parser.add_argument('--max_seqLen', default=10, type=int,
                        help='Max input sentences length.Defalut is 10')
    parser.add_argument('--droprate_p', default=0.1,
                        type=float, help='Dropout rate.Defalut is 0.1')
    parser.add_argument('--max_iters', default=75000, type=int,
                        help='Model training total iters.Default is 75000')
    parser.add_argument('--save_iters', default=5000, type=int,
                        help='Model in save each iters.Default is 5000')
    parser.add_argument('--model_dir', default='.\MT_model_fra2eng',
                        type=str, help='Your Model want to saved into dir of model_path')
    parser.add_argument('--input_lang', default='eng',
                        type=str, help='You want to translate lang')
    parser.add_argument('--trans_lang', default='fra', type=str,
                        help='lang which is translated from input_lang')
    parser.add_argument('--SOS_tokenId', default=0, type=int,
                        help='Seq special token id for sequence start.Defalut is 0')
    parser.add_argument('--EOS_tokenId', default=1, type=int,
                        help='Seq special token id for sequence ending.Defalut is 1')
    parser.add_argument('--init_prob', default=1.0, type=float,
                        help='set schedula sampling init prob.Default is 1.0')
    parser.add_argument('--do_train', action='store_true',
                        help='Required Model to training.')
    parser.add_argument('--do_eval', action='store_true',
                        help='Required Model to evluate.')
    args = parser.parse_args()

    SOS_token = args.SOS_tokenId
    EOS_token = args.EOS_tokenId
    device = torch.device(
        'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    init_logger()
    main(args)
