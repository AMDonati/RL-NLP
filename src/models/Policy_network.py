# see tutorial for Image Captioning in Tensorflow.
# Use a CNN Encoder to encode the pre-trained features.

'''LSTM Policy Network taking as input multi-model data (img_features, words embeddings'''
import torch
import torch.nn as nn
import torch.nn.functional as F
import h5py
import numpy as np


class PolicyLSTM(nn.Module):
    def __init__(self, num_tokens, word_emb_size, emb_size, hidden_size, num_filters=None, num_layers=1, p_drop=0,
                 pooling=True, value_fn=False):
        super(PolicyLSTM, self).__init__()
        self.num_tokens = num_tokens
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_drop = p_drop
        if num_filters is None:
            self.num_filters = word_emb_size
        else:
            self.num_filters = num_filters #TODO add a min between 64 and word_emb_size.
        self.pooling = pooling
        self.dropout = nn.Dropout(p_drop)
        self.embedding = nn.Embedding(num_tokens, word_emb_size)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=1)
        if pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.lstm = nn.LSTM(input_size=emb_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                dropout=p_drop,
                                batch_first=True)
        self.action_head = nn.Linear(hidden_size, num_tokens)
        if value_fn:
            self.value_head = nn.Linear(hidden_size, 1)
        else:
            self.value_head = None

    def forward(self, text_inputs, img_feat):
        '''
        :param text_inputs: shape (S, B)
        :param img_feat: shape (B, C, H, W)
        :param hidden: shape (num_layers, B, hidden_size)
        :return:
        log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
        '''
        # embeddings of text input
        words_emb = self.embedding(text_inputs)  # shape (B, S, emb_size)
        seq_len = words_emb.size(1)
        words_emb = self.dropout(words_emb)
        # encoding of img features:
        img_feat = F.relu(self.conv(img_feat))
        if self.pooling:
            img_feat = self.max_pool(img_feat)
        img_feat = img_feat.view(img_feat.size(0), -1).unsqueeze(1).repeat(1, seq_len, 1)  # shape (B, S, C*H*W)
        vis_text_emb = torch.cat([words_emb, img_feat], axis=-1)
        output, hidden = self.lstm(vis_text_emb)
        output = self.dropout(output)

        logits = self.action_head(output)  # (S,B,num_tokens)
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        if self.value_head is not None:
            value = self.value_head(output)
            value = value.view(-1, 1)
        else:
            value = None

        return logits, hidden, value


class PolicyMLP(nn.Module):
    def __init__(self, num_tokens, word_emb_size, units, num_filters=None, pooling=True):
        super(PolicyMLP, self).__init__()
        self.num_tokens = num_tokens
        self.units = units
        if num_filters is None:
            self.num_filters = word_emb_size
        else:
            self.num_filters = num_filters
        self.pooling = pooling
        self.embedding = nn.Embedding(num_tokens, word_emb_size)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=1)
        if pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.fc = nn.Linear(units, num_tokens)

    def forward(self, text_inputs, img_feat):
        '''
        :param text_inputs: shape (S, B)
        :param img_feat: shape (B, C, H, W)
        :param hidden: shape (num_layers, B, hidden_size)
        :return:
        log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
        '''
        # embeddings of text input
        words_emb = self.embedding(text_inputs)  # shape (B, S, emb_size)
        seq_len = words_emb.size(1)
        # encoding of img features:
        img_feat = F.relu(self.conv(img_feat))
        if self.pooling:
            img_feat = self.max_pool(img_feat)
        img_feat = img_feat.view(img_feat.size(0), -1).unsqueeze(1).repeat(1, seq_len, 1)  # shape (B, S, C*H*W)
        vis_text_emb = torch.cat([words_emb, img_feat], axis=-1)
        logits = self.fc(vis_text_emb)
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)

        return logits


if __name__ == '__main__':
    train_features_path = '../../data/train_features.h5'
    hf = h5py.File(train_features_path, 'r')
    img_feat = hf.get('features')
    img_feat = np.array(img_feat)
    print('features shape', img_feat.shape)  # shape (num_samples, 1024, 14, 14).

    img_feat = torch.tensor(img_feat, dtype=torch.float32)
    seq_len = 10
    num_tokens = 20
    hidden_size = 128
    dummy_text_input = torch.ones(img_feat.size(0), seq_len, dtype=torch.long)

    model = PolicyLSTM(num_tokens=num_tokens,
                       word_emb_size=64,
                       emb_size=64 + 64 * 7 * 7,
                       hidden_size=128,
                       num_layers=1,
                       p_drop=0,
                       value_fn=True)

    output, hidden, value = model(dummy_text_input, img_feat)  # shape (B*S, num_tokens)
    output = output.view(-1, seq_len, num_tokens)
    print('output shape', output.shape)
    # print("sample output", output[0, :, :])

    # ------- test of PolicyMLP --------------------------------------------------------------------------------------------------------
    dummy_model = PolicyMLP(num_tokens=num_tokens,
                            word_emb_size=64,
                            units=64 + 64 * 7 * 7)
    output = dummy_model(dummy_text_input, img_feat)
    output = output.view(-1, seq_len, num_tokens)
    print('output shape', output.shape)
    # print("sample output", output[0, :, :])
