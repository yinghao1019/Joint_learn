import torch.nn as nn


class intent_classifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(intent_classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_layer = nn.Linear(input_dim, output_dim)
        self.dropout = dropout_rate

    def forward(self, hiddens):
        # forward pass
        hiddens = self.dropout(hiddens)
        return self.fc_layer(hiddens)


class slot_classifier(nn.Module):
    def __init__(self, input_dim, output_dim, dropout_rate):
        super(slot_classifier, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.fc_layer = nn.Linear(input_dim, output_dim)
        self.dropout = dropout_rate

    def forward(self, hiddens):
        # forward pass
        hiddens = self.dropout(hiddens)
        return self.fc_layer(hiddens)
