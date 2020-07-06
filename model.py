import torch.nn as nn
import torch.nn.functional as F
import torch
from torch.autograd import Variable
class rnnModel(nn.Module):
    def __init__(self,hidden_dim,feature_dim,batch_size,use_gpu=True):
            super(rnnModel, self).__init__()
            self.feature_dim=feature_dim
            self.hidden_dim=hidden_dim
            self.num_layers=1
            self.batch_size=batch_size
            self.gpu=use_gpu
            
            # rnn
            self.bigru=nn.GRU(self.feature_dim,self.hidden_dim,num_layers=self.num_layers,bidirectional=True,batch_first=True)

            # linear
            self.hidden2label = nn.Linear(self.hidden_dim * 2, 1)

            # hidden
            self.hidden = self.init_hidden()

    
    def init_hidden(self):
        if self.gpu is True:
            if isinstance(self.bigru, nn.LSTM):
                h0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                c0 = Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim).cuda())
                return h0, c0
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim)).cuda()
        else:
            return Variable(torch.zeros(self.num_layers * 2, self.batch_size, self.hidden_dim))     
    
    def forward(self, feature):

        gru_out, hidden = self.bigru(feature, self.hidden)

        gru_out = torch.transpose(gru_out, 1, 2)
        # pooling
        gru_out = F.max_pool1d(gru_out, gru_out.size(2)).squeeze(2)

        # linear
        y = self.hidden2label(gru_out)
        y=y.squeeze(1)
        return y

