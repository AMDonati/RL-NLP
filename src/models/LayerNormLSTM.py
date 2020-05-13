'''
Implementation of a LSTM with LayerNorm.
Inspired from: https://github.com/pytorch/pytorch/issues/11335
'''

import torch
import torch.nn as nn
import torch.nn.functional as F


class LayerNormLSTMCell(nn.LSTMCell):
    '''
    ~LSTMCell.weight_ih – the learnable input-hidden weights, of shape (4*hidden_size, input_size)
    ~LSTMCell.weight_hh – the learnable hidden-hidden weights, of shape (4*hidden_size, hidden_size)
    ~LSTMCell.bias_ih – the learnable input-hidden bias, of shape (4*hidden_size)
    ~LSTMCell.bias_hh – the learnable hidden-hidden bias, of shape (4*hidden_size)
    '''

    def __init__(self, input_size, hidden_size, p_drop=0, bias=True):
        super(LayerNormLSTMCell, self).__init__(input_size=input_size, hidden_size=hidden_size, bias=bias)

        self.p_drop = p_drop
        self.ln_ih = nn.LayerNorm(4 * hidden_size)
        self.ln_hh = nn.LayerNorm(4 * hidden_size)
        self.ln_ho = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p_drop)

    def forward(self, input, hidden=None):

        self.check_forward_input(input)  # check correct shape of input for forward pass.
        if hidden is None:
            # initialization of (c,h) to zeros tensors with same dtype & device than input.
            hx = input.new_zeros(input.size(0), self.hidden_size,
                                 requires_grad=False)  # hx, and cx should be of size (batch_size, hidden_size).
            cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        self.check_forward_hidden(input=input, hx=hx,
                                  hidden_label='[0]')  # check correct shape of hidden compared to input for h.
        self.check_forward_hidden(input=input, hx=cx,
                                  hidden_label='[1]')  # check correct shape of hidden compared to input for c.

        # computation of gates:
        gates = self.ln_ih(F.linear(input=input, weight=self.weight_ih, bias=self.bias_ih)) \
                + self.ln_hh(
            F.linear(input=hx, weight=self.weight_hh, bias=self.bias_hh))  # shape (batch_size, 4 * hidden_size)

        i = gates[:, :self.hidden_size].sigmoid()
        f = gates[:, self.hidden_size:2 * self.hidden_size].sigmoid()
        o = gates[:, 2 * self.hidden_size:3 * self.hidden_size].sigmoid()
        g = gates[:, 3 * self.hidden_size:].tanh()

        # update of cell gate
        cy = f * cx + i * g
        hy = o * self.ln_ho(cy).tanh()
        hy = self.dropout(hy)

        return hy, cy


class LayerNormLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=1, p_drop=0):
        super(LayerNormLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.hidden = nn.ModuleList([LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size),
                                                       hidden_size=hidden_size,
                                                       p_drop=(0. if layer == num_layers - 1 else p_drop)) for layer in
                                     range(num_layers)])
        # shape (num_layers, batch_size, hidden_size).

    def forward(self, input, hidden=None):
        batch_size, seq_len, hidden_size = input.size()
        if hidden is None:
            hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False)
            cx = input.new_zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False)
        else:
            hx, cx = hidden

        ht, ct = [], []
        h, c = hx, cx
        for t in range(seq_len):
            x = input[:, t, :]
            h_t_l, c_t_l = [], []
            for l, layer in enumerate(self.hidden):
                h_tl, c_tl = layer(input=x, hidden=(h[l], c[l]))
                x = h_tl
                h_t_l.append(h_tl)
                c_t_l.append(h_tl)
            ht.append(h_t_l)
            ct.append(c_t_l)
            h, c = ht[t], ct[t]
        output = torch.stack([h[-1] for h in ht],
                             dim=1)  # sequence of hidden states from last layer. shape (S,B,hidden_size).
        hy = torch.stack(ht[-1])  # last hidden state (of the last timestep). shape (num_layers, B, hidden_size).
        cy = torch.stack(ct[-1])

        return output, (hy, cy)
