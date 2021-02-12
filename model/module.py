import torch.nn as nn
import torch


class intent_classifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(intent_classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_layer = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hiddens, contexts=None, attn_weights=None):
        # contexts=[Bs,seqLen,hid_dim]
        # attn_weights=[Bs,seqLen]

        if (contexts is not None) and (attn_weights is not None):
            attn_weights = attn_weights.unsqueeze(1)
            # compute attention vector
            # contexts=[Bs,hid_dim]
            contexts = torch.matmul(attn_weights, contexts).squeeze(1)

            # hiddens=[Bs,hid_dim+context_dim]
            hiddens = torch.cat((hiddens, contexts), dim=1)

        hiddens = self.dropout(hiddens)
        return self.fc_layer(hiddens)


class slot_classifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(slot_classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_layer = nn.Linear(input_dim, output_dim)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, hiddens):
        # forward pass
        hiddens = self.dropout(hiddens)
        return self.fc_layer(hiddens)
