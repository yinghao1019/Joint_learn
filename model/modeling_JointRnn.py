import torch
import random
import json
import os
import utils
from torch import nn
from torch.nn import functional as F
from .module import slot_classifier, intent_classifier


class BiEncoder(nn.Module):
    def __init__(self, configs, args):
        super(BiEncoder, self).__init__()

        self.input_dim = configs.get('input_dim')
        self.embed_dim = configs.get('embed_dim')
        self.hid_dim = configs.get('hid_dim')
        self.n_layers = configs.get('n_layers')

        self.embed = nn.Embedding(self.input_dim, self.embed_dim)
        self.rnn_layer = nn.LSTM(
            self.embed_dim, self.hid_dim, self.n_layers, bidirectional=True)

    def forward(self, input_tensors):
        # embed_tensors=[seqlen,bs,embed_dim]
        embed_tensors = self.embed(input_tensors)
        encoder_output, (h_output, c_output) = self.rnn_layer(embed_tensors)

        # return encoded,final backward rnn hidden
        # encoder=[seqlen,bs,hid_dim]
        # h_output,c_output=[1,bs,hid_dim]
        return encoder_output, (h_output, c_output)


class Decoder(nn.Module):
    def __init__(self, configs, args):
        super(Decoder, self).__init__()
        self.output_dim = configs.get('slot_label_nums')
        self.embed_dim = configs.get('embed_dim')
        self.hid_dim = configs.get('hid_dim')

        self.embed = nn.Embedding(self.output_dim, self.embed_dim)
        self.decode_layer = nn.LSTM(
            self.embed_dim+self.hid_dim*4, self.hid_dim, 1)
        self.classifier = slot_classifier(
            self.hid_dim, self.output_dim, args.dropout)

    def forward(self, input_tensors, aligned_input, encoder_outputs, attn_weight, h_state, c_state):
        # get t-1 embed_vector
        # embed_tensors=[1,bs,hid_dim]
        # aligned_tensors=[1,bs,hid_dim]
        aligned_input = aligned_input.unsqueeze(0)
        embed_tensors = self.embed(input_tensors).unsqueeze(0)

        # compute context vector
        # context=[1,bs,hid_dim*2]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        context_tensors = torch.matmul(
            attn_weight.unsqueeze(1), encoder_outputs).permute(1, 0, 2)

        # concatenate tensors(aligned,embed,context)
        # inputs=[1,bs,embed_dim+aligned_dim+context_dim]
        # outputs=[1,bs,hid_dim]
        inputs = torch.cat(
            (embed_tensors, aligned_input, context_tensors), dim=2)
        outputs, (h_state, c_state) = self.decode_layer(
            inputs, (h_state, c_state))

        # outputs=[bs,output_dim]
        outputs = self.classifier(outputs.squeeze(0))

        # return logitics & hiddens state
        return outputs, h_state, c_state


class Attentioner(nn.Module):
    def __init__(self, configs):
        super(Attentioner, self).__init__()
        self.attn_layer = nn.Linear(configs.get('hid_dim')*3, 1, bias=False)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, decoder_hiddens, encoder_outputs):
        # encoder_outputs=[bs,seqlen,hid_dim]
        seqLen = encoder_outputs.shape[0]
        encoder_outputs = encoder_outputs.permute(1, 0, 2)
        # replicate decoder_hidden=[bs,seqLen,hid_dim]
        decoder_hiddens = decoder_hiddens.unsqueeze(1).repeat(1, seqLen, 1)

        # hiddens=[Bs,seqLen,hid_dim*2]
        hiddens = torch.cat((decoder_hiddens, encoder_outputs), dim=2)

        # compute attention weight=[Bs,seqLen]
        attn_weight = self.softmax(self.attn_layer(hiddens).squeeze(2))

        return attn_weight


class Joint_AttnSeq2Seq(nn.Module):
    def __init__(self, configs, args):
        super(Joint_AttnSeq2Seq, self).__init__()
        self.encoder = BiEncoder(configs, args)
        self.decoder = Decoder(configs, args)
        self.attner = Attentioner(configs)
        self.intent_classifier = intent_classifier(
            configs['hid_dim']*3, configs['intent_label_nums'], args.dropout)
        self.args = args
        self.criterion = nn.CrossEntropyLoss(ignore_index=args.ignore_index)
        self.device = torch.device('cuda:0') if torch.cuda.is_available() or not args.no_cuda \
            else torch.device('cpu')

    def forward(self, src_tensors, trg_tensors, intent_labels, teach_ratio=0.5):
        seqLen = trg_tensors.size()[0]
        Batch_size = trg_tensors.size()[1]
        slot_labels_num = self.decoder.output_dim

        # build slot_preds container
        slot_preds = torch.zeros(
            seqLen, Batch_size, slot_labels_num, device=self.device)
        total_loss = 0
        # 1.encoder stage
        # encoder output=[seqlen,bs,hid_dim]
        encoder_outputs, (h_state, c_state) = self.encoder(src_tensors)

        # get backward rnn hidden_state
        # decoder_hidden=[Bs,hid_dim]
        # aligned_input=[seqLen-1,Bs,hid_dim]
        decoder_hidden, decoder_cell = h_state[-1, :, :], c_state[-1, :, :]

        # compute attention weight
        decoder_attnW = self.attner(decoder_hidden, encoder_outputs)

        # 1.compute intent_loss
        # intent_logitics=[Bs,intent_labels_num]
        intent_logitics = self.intent_classifier(decoder_hidden, contexts=encoder_outputs.permute(1, 0, 2),
                                                 attn_weights=decoder_attnW)
        intent_loss = self.criterion(intent_logitics, intent_labels)
        total_loss += intent_loss

        # get remove <eos> input
        # trg_inputs=[seqlen-1,bs]
        trg_inputs = trg_tensors[:-1, :]

        # get decoder init_input
        # decoder_input=[Bs,]
        # decoder hidden/cell=[1,Bs,hid_dim]
        decoder_input = trg_inputs[0, :]
        decoder_hidden = decoder_hidden.unsqueeze(0)
        decoder_cell = decoder_cell.unsqueeze(0)
        # slot_decode stage
        for idx in range(1, seqLen):

            # aligned inputs=[Bs,encoder_hid_dim]
            # t+1 deocder_input=[Bs,slot_labels_num]
            aligned = encoder_outputs[idx, :, :]
            decoder_input, decoder_hidden, decoder_cell = self.decoder(decoder_input, aligned, encoder_outputs,
                                                                       decoder_attnW, decoder_hidden, decoder_cell)
            # save decoder predict
            slot_preds[idx] = decoder_input

            # compute attention weights
            decoder_attnW = self.attner(
                decoder_hidden[-1, :, :], encoder_outputs)

            # compute next decoder_token id
            # t+1 decoder_input=[Bs,]
            use_teacher = True if random.random() < teach_ratio else False
            top1_predict = torch.argmax(F.softmax(decoder_input, dim=1), dim=1)

            decoder_input = trg_tensors[idx] if use_teacher else top1_predict

        # compute slot_loss
        slot_loss = self.criterion(slot_preds[1:, :, :].reshape(-1, slot_labels_num),
                                   trg_tensors[1:, :].reshape(-1))

        total_loss += (self.args.slot_loss_coef*slot_loss)

        return (total_loss, (intent_logitics, slot_preds[1:, :, :].permute(1, 0, 2)))

    def get_predict(self, src_tensors, origin_len, trg_initTokenId, trg_endTokenId):
        slot_preds = []
        # 1.encoder stage
        # encoder output=[seqlen,bs,hid_dim]
        encoder_outputs, (h_state, c_state) = self.encoder(src_tensors)
        aligned_inputs = encoder_outputs[1:, :, :]
        # get backward rnn hidden_state
        # decoder_hidden=[Bs,hid_dim]
        # aligned_output=[seqLen,Bs,hid_dim]
        decoder_hidden, decoder_cell = h_state[-1, :, :], c_state[-1, :, :]

        # compute attention weight
        decoder_attnW = self.attner(decoder_hidden, encoder_outputs)
        # get intent logticis
        # intent_logitics=[Bs,intent_labels_num]
        intent_logitics = self.intent_classifier(
            decoder_hidden, contexts=encoder_outputs.permute(1, 0, 2), attn_weights=decoder_attnW)

        # get decoder logitics
        decoder_input = torch.tensor([trg_initTokenId], device=self.device)
        align_maxLen = origin_len
        decoder_hidden = decoder_hidden.unsqueeze(0)
        decoder_cell = decoder_cell.unsqueeze(0)

        for idx in range(align_maxLen):
            # aligned=[Bs,]
            aligned = aligned_inputs[idx]

            # decoder_input=[Bs,slot_dim]
            # deocder_hidden=[num_layer*direction,Bs,hid_dim]
            # decoder_cell=[num_layer*direction,Bs,hid_dim]
            decoder_input, decoder_hidden, decoder_cell = self.decoder(decoder_input, aligned, encoder_outputs,
                                                                       decoder_attnW, decoder_hidden, decoder_cell)
            # compute attention weights
            decoder_attnW = self.attner(
                decoder_hidden[-1, :, :], encoder_outputs)

            # get next decoder input
            decoder_input = torch.argmax(
                F.softmax(decoder_input, dim=1), dim=1)

            slot_preds.append(decoder_input.cpu().item())

            if decoder_input.item() == trg_endTokenId:
                break
        return intent_logitics, slot_preds

    @classmethod
    def reload_model(cls, model_dir_path, train_args):
        # confirm whether model dir exists
        if not os.path.exists(model_dir_path):
            raise FileNotFoundError('Model_dir_path not exists')
        config_path = os.path.join(model_dir_path, 'config.json')
        params_path = os.path.join(model_dir_path, 'pretrain_model.pt')

        # loading  data
        with open(config_path, 'r', encoding='utf-8') as f_r:
            configs = json.load(f_r)

        model_params = torch.load(params_path)['model_state_dict']
        # initalize model object

        model = cls(configs, train_args)

        # loading pretrained weights
        model.load_state_dict(model_params)

        return model


RnnConfig = {
    'input_dim': None,
    'embed_dim': 512,
    'hid_dim': 256,
    'n_layers': 1,
    'slot_label_nums': None,
    'intent_label_nums': None,
}
