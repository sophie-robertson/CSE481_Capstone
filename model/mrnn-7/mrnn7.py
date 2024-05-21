import torch
from torch import nn

class MilliesRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, bidirc=False):
        super(MilliesRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size

        self.bidirc = bidirc
        if self.bidirc:
            self.hidden_size += self.output_size

        # visual cortex
        self.i2h = nn.Linear(self.input_size + self.output_size, 100)
        self.h2h = nn.Linear(100, 100)
        self.h2o = nn.Linear(100, 100)

        # striatum
        self.stri = nn.Linear(151, 50)

        # gpe & gpi
        #   GPE could be modeled as an RNN with the STN ?
        self.gpe = nn.Linear(51, 50)
        self.gpi = nn.Linear(101, 50)


        # thalamus
        self.thal = nn.Linear(101, self.output_size)

        # motor cortex
        self.i2h_dos = nn.Linear(151, self.hidden_size)
        self.h2h_dos = nn.Linear(self.hidden_size, self.hidden_size)
        self.h2o_dos = nn.Linear(self.hidden_size, self.output_size)
        
        

    def forward(self, data, dopamine = 1): 
        """
        x --> N x L
        hidden state --> N x H

        output --> N x O
        """
        images = data[:, :, :-1]
        holds = data[:, :, -1].reshape(data.shape[0], data.shape[1], 1)
        batch_size = data.shape[0]
        trial_len = data.shape[1]

        outputs_v = torch.zeros((batch_size, trial_len, 100))
        outputs_s = torch.zeros((batch_size, trial_len, 50))
        outputs_gpe = torch.zeros((batch_size, trial_len, 50))
        outputs_gpi = torch.zeros((batch_size, trial_len, 50))
        outputs_m = torch.zeros((batch_size, trial_len, self.output_size))
        outputs_t = torch.zeros((batch_size, trial_len, self.output_size))
        hidden_states_v = torch.zeros((batch_size, trial_len, 100))
        hidden_states_m = torch.zeros((batch_size, trial_len, self.hidden_size))
        hidden_state_v = self.init_hidden(batch_size, 100)
        hidden_state_m = self.init_hidden(batch_size, 150)

        snc = self.init_snc(batch_size, dopamine) # N x 50

        inp_mots = torch.zeros((batch_size, trial_len, self.output_size))

        
        for i in range(trial_len):
            image = images[:, i, :]
            hold = holds[:, i, :]

            # visual cortex
            if self.bidirc: 
                inp_vis = torch.cat((image, outputs_m[:, i-1,:]), dim=1)
            inp_vis = torch.cat((inp_vis, hold), dim=1)
            inp_vis = self.i2h(inp_vis) # N x H
            hidden_state_v = self.h2h(hidden_state_v)
            hidden_state_v = self.retanh(inp_vis + hidden_state_v)
            out_v = self.h2o(hidden_state_v)
            out_v = self.retanh(out_v)
            outputs_v[:, i, :] = out_v
            hidden_states_v[:, i, :] = hidden_state_v

            # striatum
            inp_stri = torch.cat((out_v, hold), dim=1)
            inp_stri = torch.cat((inp_stri, snc), dim=1)
            out_stri = self.retanh(self.stri(inp_stri))
            outputs_s[:, i, :] = out_stri

            # gpe & gpi
            inp_gpe = torch.cat((out_stri, hold), dim=1)
            out_gpe = self.retanh(self.gpe(inp_gpe))
            outputs_gpe[:, i, :] = out_gpe

            inp_gpi = torch.cat((out_gpe, out_stri), dim=1)
            inp_gpi = torch.cat((inp_gpi, hold),  dim=1)
            out_gpi = self.retanh(self.gpi(inp_gpi))
            outputs_gpi[:, i, :] = out_gpi

            # thalamus
            if self.bidirc:
                inp_thal = torch.cat((out_gpi, outputs_m[:, i-1, :]), dim=1)
            inp_thal = torch.cat((inp_thal, hold), dim=1)
            out_thal = self.retanh(self.thal(inp_thal))
            outputs_t[:, i, :] = out_thal

            # motor cortex
            if self.bidirc:
                inp_mot = torch.cat((out_thal, outputs_v[:, i-1, :]), dim=1)
            inp_mot = torch.cat((inp_mot, hold), dim=1)
            
            inp_mot = self.i2h_dos(inp_mot) 
            hidden_state_m = self.h2h_dos(hidden_state_m)
            hidden_state_m = self.retanh(inp_mot + hidden_state_m)
            out_m = self.h2o_dos(hidden_state_m)
            outputs_m[:, i, :] = out_m
            hidden_states_m[:, i, :] = hidden_state_m

        return outputs_m

    
    def init_hidden(self, batch_size, hidden_size):
        # return nn.init.kaiming_uniform_(torch.empty(self.num_layers, self.hid_dim))
        return nn.init.kaiming_uniform_(torch.empty(batch_size, hidden_size)) # for bidirectional
    
    def retanh(self, x):
        return torch.tanh(torch.clamp(x, min=0))
    
    def init_snc(self, batch_size, dope):
        return torch.nn.init.normal_(torch.empty(batch_size, 50), mean=dope, std=1.0)