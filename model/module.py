import torch.nn as nn
class IntentClassifier(nn.Module):
    def __init__(self,input_dim,output_dim,dropout_r=0.1):
        super(IntentClassifier,self).__init__()
        self.linear_layer=nn.Linear(input_dim,output_dim)
        self.dropout=nn.Dropout(dropout_r)
    def forward(self,input_features):
        x=self.dropout(input_features)
        return self.linear_layer(x)
class SlotClassifier(nn.Module):
    def __init__(self,input_dim,output_dim,dropout_r=0.1):
        super(SlotClassifier,self).__init__()
        self.Linear=nn.Linear(input_dim,output_dim)
        self.dropout=nn.Dropout(dropout_r)
    def forward(self,inputs_feature):
        x=self.dropout(inputs_feature)
        return self.Linear(x)