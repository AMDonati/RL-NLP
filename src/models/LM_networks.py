# code inspired from: https://github.com/pytorch/examples/blob/master/word_language_model/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class GRUModel(nn.Module):

  def __init__(self, num_tokens, emb_size, hidden_size, num_layers=1, p_drop=0):
    super(GRUModel, self).__init__()
    self.num_tokens = num_tokens
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.p_drop = p_drop

    self.dropout = nn.Dropout(p_drop)
    self.embedding = nn.Embedding(num_tokens, emb_size)
    self.gru = nn.GRU(input_size=emb_size, hidden_size=hidden_size, num_layers=num_layers, dropout=p_drop)
    self.fc = nn.Linear(in_features=hidden_size, out_features=num_tokens)

    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.fc.weight.data.uniform_(-initrange, initrange)
    self.fc.bias.data.zero_()

  def forward(self, input, hidden):
    emb = self.embedding(input)
    emb = self.dropout(emb)
    output, hidden = self.gru(emb, hidden) # (seq_len, batch_size, hidden_size)
    output = self.dropout(output)
    dec_output = self.fc(output) # (seq_len, batch_size, num_tokens)
    dec_output = dec_output.view(-1, self.num_tokens) # (seq_len * batch_size, num_tokens)
    log_probas = F.log_softmax(dec_output, dim=1)

    return log_probas, hidden

  def init_hidden(self, batch_size):
    weight = next(self.parameters()) #TODO understand this part.
    return weight.new_zeros(self.num_layers, batch_size, self.hidden_size)

if __name__ == '__main__':
    batch_size = 8
    emb_size = 512
    hidden_size = 128
    num_tokens = 85
    seq_len = 20
    device = torch.device("cpu")
    inputs = torch.ones(seq_len, batch_size, dtype=torch.long).to(device)
    model = GRUModel(num_tokens=num_tokens, emb_size=emb_size, hidden_size=hidden_size)
    hidden = model.init_hidden(batch_size)
    output, hidden = model(inputs, hidden)
    print('output', output.shape)
    print('hidden', hidden.shape)
    #output = output.view(batch_size, seq_len, num_tokens)
    #print('output reshaped', output.shape)