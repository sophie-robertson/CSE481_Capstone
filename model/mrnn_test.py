import torch
from torch import nn

class MilliesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(MilliesRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size

        self.prev_output = None
        self.x = None

        self.x0 = nn.Parameter(torch.empty(self.num_neurons))
        self.J = 


        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        self.h2o = nn.Linear(hidden_size, output_size)
        
        

    def forward(self, data):
        """
        data: 
            - hold: either 0.0 or 1.0
            - image: L x N (21 x batch_size)

        output --> N x O
        """
        image, hold = data

        if self.prev_output is None: # initializing hidden state it not there?
            self.x = torch.tile(self.x0, (self.batch_size, 1))
            self.prev_output = self.retanh(self.x)

        recur = self.prev_output @ self.J.T # hidden space to hidden space

        inp = image.T @ self.I.T + hold.T * self.S.T # input to hidden space plus hold

        


        x = self.i2h(x) # input to hidden space
        hidden_state = self.h2h(hidden_state) # hidden space to hidden space
        hidden_state = self.retanh(x+hidden_state) 
        out = self.h2o(hidden_state)
        return out, hidden_state
    
    def init_hidden(self):
        # return nn.init.kaiming_uniform_(torch.empty(self.num_layers, self.hid_dim))
        return nn.init.kaiming_uniform_(torch.empty(self.batch_size, self.hidden_size)) # for bidirectional
    
    def retanh(self, x):
        return torch.tanh(torch.clamp(x, min=0))