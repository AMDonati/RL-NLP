'''LSTM Policy Network taking as input multi-model data (img_features, words embeddings'''

import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np

class PolicyLSTM(nn.Module):
  def __init__(self, num_tokens, word_emb_size, emb_size, hidden_size, num_layers, p_drop):
    super(PolicyLSTM, self).__init__()
    self.num_tokens = num_tokens
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.p_drop = p_drop
    self.dropout = nn.Dropout(p_drop)
    self.embedding = nn.Embedding(num_tokens, word_emb_size)
    self.lstm = nn.LSTM(input_size=emb_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=p_drop)
    self.fc = nn.Linear(hidden_size, num_tokens)

    self.init_weights()

  def init_weights(self):
    initrange = 0.1
    self.embedding.weight.data.uniform_(-initrange, initrange)
    self.fc.weight.data.uniform_(-initrange, initrange)
    self.fc.bias.data.zero_()


  def forward(self, text_inputs, img_feat, hidden):
    '''
    :param text_inputs: shape (S, B, num_tokens)
    :param img_feat: shape (B, C, H, W)
    :param hidden: shape (num_layers, B, hidden_size)
    :return:
    log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
    '''
    #TODO: add additionnal encoding of img features?
    words_emb = self.embedding(text_inputs) # shape (S, B, emb_size)
    seq_len = words_emb.size(0)
    words_emb = self.dropout(words_emb)
    img_feat = img_feat.view(img_feat.size(0), -1).unsqueeze(0).repeat(seq_len, 1, 1) # shape (S, B, C*H*W)
    emb = torch.cat([words_emb, img_feat], axis=-1)
    output, hidden = self.lstm(emb, hidden)
    output = self.dropout(output)
    dec_output = self.fc(output) # (S,B,num_tokens)
    dec_output = dec_output.view(-1, self.num_tokens) # (S*B, num_tokens)
    log_probas = F.log_softmax(dec_output, dim=-1)

    return log_probas, hidden

  def init_hidden(self, batch_size):
    weight = next(self.parameters())
    return (weight.new_zeros(self.num_layers, batch_size, self.hidden_size),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_size))


if __name__ == '__main__':
    train_features_path = '../../data/train_features.h5'
    hf = h5py.File(train_features_path, 'r')
    img_feat = hf.get('features')
    img_feat = np.array(img_feat)
    print('features shape', img_feat.shape) # shape (num_samples, 1024, 14, 14).

    img_feat = torch.tensor(img_feat, dtype=torch.float32)
    seq_len = 20
    num_tokens = 50
    hidden_size = 128
    dummy_text_input = torch.ones(seq_len, img_feat.size(0), dtype=torch.long)

    model = PolicyLSTM(num_tokens=num_tokens,
                       word_emb_size=64,
                       emb_size=(64 + img_feat.size(1)*img_feat.size(2)*img_feat.size(3)),
                       hidden_size=128,
                       num_layers=1,
                       p_drop=0)
    hidden = model.init_hidden(batch_size=img_feat.size(0))
    output, hidden = model(dummy_text_input, img_feat, hidden)
