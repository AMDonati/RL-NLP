# see tutorial for Image Captioning in Tensorflow.
# Use a CNN Encoder to encode the pre-trained features.

'''LSTM Policy Network taking as input multi-model data (img_features, words embeddings'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np

class PolicyLSTM(nn.Module):
  def __init__(self, num_tokens, word_emb_size, vis_emb_size, emb_size, hidden_size, num_layers, p_drop):
    super(PolicyLSTM, self).__init__()
    self.num_tokens = num_tokens
    self.emb_size = emb_size
    self.hidden_size = hidden_size
    self.num_layers = num_layers
    self.p_drop = p_drop
    self.dropout = nn.Dropout(p_drop)
    self.embedding = nn.Embedding(num_tokens, word_emb_size)
    self.fc_emb = nn.Linear(vis_emb_size, emb_size)
    self.lstm = nn.LSTM(input_size=emb_size,
                        hidden_size=hidden_size,
                        num_layers=num_layers,
                        dropout=p_drop,
                        batch_first=True)
    self.fc = nn.Linear(hidden_size, num_tokens)


  def forward(self, text_inputs, img_feat):
    '''
    :param text_inputs: shape (S, B)
    :param img_feat: shape (B, C, H, W)
    :param hidden: shape (num_layers, B, hidden_size)
    :return:
    log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
    '''
    #TODO: add additionnal encoding of img features?
    words_emb = self.embedding(text_inputs) # shape (B, S, emb_size)
    seq_len = words_emb.size(1)
    words_emb = self.dropout(words_emb)
    img_feat = img_feat.view(img_feat.size(0), -1).unsqueeze(1).repeat(1, seq_len, 1) # shape (B, S, C*H*W)
    vistext_emb = torch.cat([words_emb, img_feat], axis=-1)
    vistext_emb = F.relu(self.fc_emb(vistext_emb))
    output, hidden = self.lstm(vistext_emb)
    output = self.dropout(output)
    logits = self.fc(output) # (S,B,num_tokens)
    logits = logits.view(-1, self.num_tokens) #(S*B, num_tokens)

    return logits, hidden


if __name__ == '__main__':
    train_features_path = '../../data/train_features.h5'
    hf = h5py.File(train_features_path, 'r')
    img_feat = hf.get('features')
    img_feat = np.array(img_feat)
    print('features shape', img_feat.shape) # shape (num_samples, 1024, 14, 14).

    img_feat = torch.tensor(img_feat, dtype=torch.float32)
    seq_len = 10
    num_tokens = 20
    hidden_size = 128
    dummy_text_input = torch.ones(img_feat.size(0), seq_len, dtype=torch.long)

    model = PolicyLSTM(num_tokens=num_tokens,
                       word_emb_size=64,
                       vis_emb_size=64 + img_feat.size(1) * img_feat.size(2) * img_feat.size(3),
                       emb_size=128,
                       hidden_size=128,
                       num_layers=1,
                       p_drop=0)

    output, hidden = model(dummy_text_input, img_feat) # shape (B*S, num_tokens)
    output = output.view(-1, seq_len, num_tokens)
    print("sample output", output[:5, :, :])
