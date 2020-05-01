# code inspired from: https://github.com/pytorch/examples/blob/master/word_language_model/model.py
#TODO - Add layer norm: https://pytorch.org/docs/stable/nn.html?highlight=layernorm#torch.nn.LayerNorm


import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUModel(nn.Module):

  def __init__(self, num_tokens, emb_size, hidden_size, num_layers=1, p_drop=0, bidirectional = False):
    super(GRUModel, self).__init__()
    self.num_tokens = num_tokens
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.p_drop = p_drop
    self.dropout = nn.Dropout(p_drop)
    self.bidirectional = bidirectional
    self.embedding = nn.Embedding(num_tokens, emb_size)
    self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, dropout=p_drop, bidirectional=bidirectional)
    if bidirectional:
      in_features = 2 * hidden_size
    else:
      in_features = hidden_size
    self.fc = nn.Linear(in_features=in_features, out_features=num_tokens)

    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.fc.weight.data.uniform_(-initrange, initrange)
    self.fc.bias.data.zero_()

  def forward(self, input, hidden):
    emb = self.embedding(input)
    emb = self.dropout(emb)
    output, hidden = self.gru(emb, hidden) # # output (S,B,hidden_size*num_directions) # hidden: (num_layers * num_directions, B, hidden_size)
    output = self.dropout(output)
    dec_output = self.fc(output) # (S, B, num_tokens)
    dec_output = dec_output.view(-1, self.num_tokens) # (S * B, num_tokens)
    log_probas = F.log_softmax(dec_output, dim=1)

    return log_probas, hidden


  def init_hidden(self, batch_size):
    weight = next(self.parameters()) #TODO understand this part.
    self.bidirectional = False
    if self.bidirectional:
      return weight.new_zeros(self.num_layers*2, batch_size, self.hidden_size)
    else:
      return weight.new_zeros(self.num_layers, batch_size, self.hidden_size)

class LSTMModel(nn.Module):
  def __init__(self, num_tokens, emb_size, hidden_size, num_layers=1, p_drop=0, bidirectional=False):
    super(LSTMModel, self).__init__()
    self.num_tokens = num_tokens
    self.emb_size = emb_size
    self.num_tokens = num_tokens
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.p_drop = p_drop
    self.bidirectional = bidirectional
    self.dropout = nn.Dropout(p_drop)
    self.embedding = nn.Embedding(num_tokens, emb_size)
    self.lstm = nn.LSTM(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, dropout=p_drop)
    if bidirectional:
      in_features = 2 * hidden_size
    else:
      in_features = hidden_size
    self.fc = nn.Linear(in_features=in_features, out_features=num_tokens)

    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.fc.weight.data.uniform_(-initrange, initrange)
    self.fc.bias.data.zero_()

  def forward(self, input, hidden):
    emb = self.embedding(input)
    emb = self.dropout(emb)
    output, hidden = self.lstm(emb, hidden) # output (S,B,hidden_size*num_dimension) # hidden: (num_layers * num_directions, B, hidden_size)
    output = self.dropout(output)
    dec_output = self.fc(output) # (S,B,num_tokens)
    dec_output = dec_output.view(-1, self.num_tokens) # (S*B, num_tokens)
    log_probas = F.log_softmax(dec_output, dim=1)

    return log_probas, hidden

  def init_hidden(self, batch_size):
    weight = next(self.parameters())
    if self.bidirectional:
      return (weight.new_zeros(self.num_layers*2, batch_size, self.hidden_size),
              weight.new_zeros(self.num_layers*2, batch_size, self.hidden_size))
    else:
      return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_size))

if __name__ == '__main__':
    batch_size = 8
    emb_size = 512
    hidden_size = 128
    num_tokens = 85
    seq_len = 20
    device = torch.device("cpu")
    inputs = torch.ones(seq_len, batch_size, dtype=torch.long).to(device)
    inputs_temp = inputs[:, [1,5,7]]
    model = GRUModel(num_tokens=num_tokens, emb_size=emb_size, hidden_size=hidden_size, bidirectional=True)
    hidden = model.init_hidden(batch_size)
    output, hidden = model(inputs, hidden)
    print('output', output.shape)
    print('hidden', hidden.shape)
    output = output.view(batch_size, seq_len, num_tokens)
    print('output reshaped', output.shape)