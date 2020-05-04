'''
Implementation of a LSTM with LayerNorm.
Inspired from: https://github.com/pytorch/pytorch/issues/11335
'''

import torch.nn as nn
import torch.nn.functional as F

class LayerNormLSTMCell(nn.LSTMCell):
  '''
  ~LSTMCell.weight_ih – the learnable input-hidden weights, of shape (4*hidden_size, input_size)
  ~LSTMCell.weight_hh – the learnable hidden-hidden weights, of shape (4*hidden_size, hidden_size)
  ~LSTMCell.bias_ih – the learnable input-hidden bias, of shape (4*hidden_size)
  ~LSTMCell.bias_hh – the learnable hidden-hidden bias, of shape (4*hidden_size)
  '''
  def __init__(self, input_size, hidden_size, bias=True):
    super(LayerNormLSTMCell, self).__init__(input_size=input_size, hidden_size=hidden_size, bias=bias)

    self.ln_ih = nn.LayerNorm(4*hidden_size)
    self.ln_hh = nn.LayerNorm(4*hidden_size)
    self.ln_ho = nn.LayerNorm(hidden_size)

  def forward(self, input, hidden=None):

    self.check_forward_input(input) # check correct shape of input for forward pass.
    if hidden is None:
      # initialization of (c,h) to zeros tensors with same dtype & device than input.
      hx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
      cx = input.new_zeros(input.size(0), self.hidden_size, requires_grad=False)
    else:
      hx, cx = hidden

    self.check_forward_hidden(input=input, hx=hx, hidden_label='[0]') # check correct shape of hidden compared to input for h.
    self.check_forward_hidden(input=input, hx=cx, hidden_label='[1]') # check correct shape of hidden compared to input for c.

    # computation of gates:
    gates = self.ln_ih(F.linear(input=input, weight=self.weight_ih, bias=self.bias_ih)) \
            + self.ln_hh(F.linear(input=hx, weight=self.weight_hh, bias=self.bias_hh)) # shape (batch_size, 4 * hidden_size)

    i = gates[:, :self.hidden_size].sigmoid()
    f = gates[:, self.hidden_size:2*self.hidden_size].sigmoid()
    o = gates[:, 2*self.hidden_size:3*self.hidden_size].sigmoid()
    g = gates[:, 3*self.hidden_size:].tanh()

    # update of cell gate
    cy = f * cx + i * g
    hy = o * self.ln_ho(cy).tanh()

    return hy, cy

class LayerNormLSTM(nn.Module):
  def __init__(self, input_size, hidden_size, num_layers=1):
    super(LayerNormLSTM, self).__init__()
    self.input_size = input_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers

    self.hidden = nn.ModuleList([LayerNormLSTMCell(input_size=(input_size if layer == 0 else hidden_size),
                                                   hidden_size=hidden_size) for layer in range(num_layers)])

  def forward(self, input, hidden=None):
    seq_len, batch_size, hidden_size = input.size()
    if hidden is None:
      hx = input.new_zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False)
      cx = input.new_zeros(self.num_layers, batch_size, self.hidden_size, requires_grad=False)
    else:
      hx, cx = hidden

    ht = input.new_zeros(seq_len, self.num_layers, batch_size, self.hidden_size)
    ct = input.new_zeros(seq_len, self.num_layers, batch_size, self.hidden_size)

    h, c = hx, cx
    for t, x in enumerate(input):
      for l, layer in enumerate(self.hidden):
        ht[t, l, :, :], ct[t, l, :, :] = layer(input=x, hidden=(h[l, :, :], c[l, :, :])) # output of cell: (B, hidden_size)
        x = ht[t, l, :, :]
      h, c = ht[t, :, :, :], ct[t, :, :, :]
    output = ht[:, -1, :, :] # shape (S, B, H) sequence of hidden states from last layer.
    hy = ht[-1, :, :, :] # last hidden state (of last timestep).
    cy = ct[-1, :, :, :] # last cell state (of last tiemstep).

    return output, (hy, cy)