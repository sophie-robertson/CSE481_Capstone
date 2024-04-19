import torch
from torch import nn

class MilliesRNN(nn.Module):
    def __init__(self, input_dim, hid_dim, num_layers, output_dim bidirc=True):
        self.test = nn.RNN(input_dim, hid_dim, num_layers, 'relu', bidirectional=bidirc)
        self.final = nn.Linear(input_dim, output_dim)

    def forward(self, data): 
        visual, hidden = data
        inter = self.test(visual, hidden)
        output = self.final(inter)

        return output