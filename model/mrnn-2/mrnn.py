import torch
from torch import nn

class MilliesRNN(nn.Module):
    def __init__(self, _input_dim, _hid_dim, _num_layers, _output_dim, bidirc=True):
        super(MilliesRNN, self).__init__()
        self.hid_dim = _hid_dim
        self.num_layers = _num_layers
        self.test = nn.RNN(_input_dim, _hid_dim, num_layers = 3, nonlinearity='tanh', bidirectional=True)
        # self.final = nn.Linear(_hid_dim, _output_dim)
        self.final = nn.Linear(2*_hid_dim, _output_dim)

    def forward(self, data): 
        inter, hn = self.test(data, self.init_hidden())
        output = self.final(inter)

        return output, hn
    
    def init_hidden(self):
        # return nn.init.kaiming_uniform_(torch.empty(self.num_layers, self.hid_dim))
        return nn.init.kaiming_uniform_(torch.empty(2*self.num_layers, self.hid_dim)) # for bidirectional