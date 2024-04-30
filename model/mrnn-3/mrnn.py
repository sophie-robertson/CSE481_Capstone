import torch
from torch import nn

class MilliesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(MilliesRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size


        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        self.h2o = nn.Linear(hidden_size, output_size)
        
        

    def forward(self, x, hidden_state): 
        """
        x --> N x L
        hidden state --> N x H

        output --> N x O
        """
        x = self.i2h(x) 
        hidden_state = self.h2h(hidden_state)
        hidden_state = self.retanh(x + hidden_state)
        out = self.h2o(hidden_state)
        return out, hidden_state
    
    def init_hidden(self):
        # return nn.init.kaiming_uniform_(torch.empty(self.num_layers, self.hid_dim))
        return nn.init.kaiming_uniform_(torch.empty(self.batch_size, self.hidden_size)) # for bidirectional
    
    def retanh(self, x):
        return torch.tanh(torch.clamp(x, min=0))