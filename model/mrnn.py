import torch
from torch import nn

class MilliesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MilliesRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.i2h = nn.Linear(input_size, hidden_size)
        self.h2h = nn.Linear(hidden_size, hidden_size)

        self.h2o = nn.Linear(hidden_size, output_size)

        self.thal = nn.Linear(output_size, output_size)

        self.i2h_dos = nn.Linear(output_size, hidden_size)
        self.h2h_dos = nn.Linear(hidden_size, hidden_size)

        self.h2o_dos = nn.Linear(hidden_size, output_size)
        
        

    def forward(self, data): 
        """
        x --> N x L
        hidden state --> N x H

        output --> N x O
        """
        batch_size = data.shape[0]

        trial_len = data.shape[1]
        outputs_v = torch.zeros((data.shape[0], data.shape[1], self.output_size))
        outputs_m = torch.zeros((data.shape[0], data.shape[1], self.output_size))
        hidden_states_v = torch.zeros((data.shape[0], data.shape[1], self.hidden_size))
        hidden_states_m = torch.zeros((data.shape[0], data.shape[1], self.hidden_size))
        hidden_state = self.init_hidden(batch_size)

        # visual cortex
        for i in range(trial_len):
            x = data[:, i, :]
            x = self.i2h(x) 
            hidden_state = self.h2h(hidden_state)
            hidden_state = self.retanh(x + hidden_state)
            out = self.h2o(hidden_state)
            outputs_v[:, i, :] = out
            hidden_states_v[:, i, :] = hidden_state

        # thalamus
        # print(outputs.shape)
        outputs_t = self.retanh(self.thal(outputs_v))

        # motor cortex
        hidden_state = self.init_hidden(batch_size)
        for i in range(trial_len):
            x = outputs_t[:, i, :]
            x = self.i2h_dos(x) 
            hidden_state = self.h2h_dos(hidden_state)
            hidden_state = self.retanh(x + hidden_state)
            out = self.h2o_dos(hidden_state)
            outputs_m[:, i, :] = out
            hidden_states_m[:, i, :] = hidden_state

        return outputs_m

    
    def init_hidden(self, batch_size):
        # return nn.init.kaiming_uniform_(torch.empty(self.num_layers, self.hid_dim))
        return nn.init.kaiming_uniform_(torch.empty(batch_size, self.hidden_size)) # for bidirectional
    
    def retanh(self, x):
        return torch.tanh(torch.clamp(x, min=0))