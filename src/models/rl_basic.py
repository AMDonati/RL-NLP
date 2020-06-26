import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence


class PolicyGRU(nn.Module):
    def __init__(self, num_tokens, word_emb_size, emb_size, hidden_size, num_filters=None, num_layers=1, p_drop=0,
                 pooling=True, cat_size=64 + 7 * 7 * 32):
        super(PolicyGRU, self).__init__()
        self.num_tokens = num_tokens
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_drop = p_drop
        if num_filters is None:
            self.num_filters = word_emb_size
        else:
            self.num_filters = num_filters
        self.pooling = pooling
        self.dropout = nn.Dropout(p_drop)
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.gru = nn.GRU(word_emb_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(cat_size, num_tokens)
        self.saved_log_probs = []
        self.rewards = []

        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=1)
        if pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.last_policy = []

    def forward(self, text_inputs, img_feat):
        '''
        :param text_inputs: shape (S, B)
        :param img_feat: shape (B, C, H, W)
        :param hidden: shape (num_layers, B, hidden_size)
        :return:
        log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
        '''
        embed_text = self._get_embed_text(text_inputs)

        img_feat = F.relu(self.conv(img_feat))
        if self.pooling:
            img_feat = self.max_pool(img_feat)

        img_feat = img_feat.view(img_feat.size(0), -1)

        embedding = torch.cat((img_feat, embed_text.view(embed_text.size(0), -1)), dim=1)
        logits = self.fc(embedding)  # (S,B,num_tokens)
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        probs = F.softmax(logits, dim=1)
        # sumprobs = probs.sum().detach().numpy()
        # if math.isnan(sumprobs):
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        # self.last_policy.append(probs.detach().numpy()[0])
        return policy_dist

    def _get_embed_text(self, text):
        _, hidden = self.gru(self.word_embedding(text))
        return hidden[-1]


class PolicyGRUWord(nn.Module):

    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1):
        super(PolicyGRUWord, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.gru = nn.GRU(word_emb_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_tokens + 1)
        self.saved_log_probs = []
        self.rewards = []
        self.values = []
        self.last_policy = []
        self.optimizer = None

    def forward(self, text_inputs, img_feat, valid_actions):
        embed_text = self._get_embed_text(text_inputs)
        out = self.fc(embed_text)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        logits = logits[:, valid_actions]
        probs = F.softmax(logits, dim=1)
        policy_dist = Categorical(probs)
        probs_ = policy_dist.probs.clone()
        # self.last_policy.append(probs_.detach().numpy()[0])
        return policy_dist, value

    def _get_embed_text(self, text):
        _, hidden = self.gru(self.word_embedding(text))
        return hidden[-1]


class PolicyLSTMWordBatch(nn.Module):

    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1, kernel_size=1, stride=5, num_filters=3, rl=True):
        super(PolicyLSTMWordBatch, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.lstm = nn.LSTM(word_emb_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_tokens + 1)
        self.saved_log_probs = []
        self.rewards = []
        self.values = []
        self.last_policy = []
        self.optimizer = None
        self.rl = rl

    def forward(self):
        raise NotImplementedError

    def act(self, state_text, state_img, valid_actions=None):
        # text_inputs, img_feat=state.text, state.img
        # states_=[torch.cat((state_.img,state_.text.view(state_.text.size(0),-1)), dim=1) for state_ in state]
        texts = [state_.text[0] for state_ in state]
        # img_feat = torch.cat([state_.img for state_ in state])

        embed_text = self._get_embed_text(texts)

        # img_feat_ = F.relu(self.conv(img_feat))
        # img_feat = img_feat_.view(img_feat.size(0), -1)

        # embedding = torch.cat((img_feat, embed_text.view(embed_text.size(0), -1)), dim=1)
        out = self.fc(embed_text)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]

        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        if valid_actions is not None:
            logits = torch.gather(logits, 1, valid_actions)
            # logits = logits[:, valid_actions]
        probs = F.softmax(logits, dim=1)
        # sumprobs = probs.sum().detach().numpy()
        # if math.isnan(sumprobs):
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        # self.last_policy.append(probs.detach().cpu().numpy()[0])
        if self.rl:
            return policy_dist, value
        else:
            return logits, value

    def _get_embed_text(self, text):
        # padded = pad_sequence(text, batch_first=True, padding_value=0).to(self.device)
        lens = (text != 0).sum(dim=1)

        pad_embed = self.word_embedding(text)
        pad_embed_pack = pack_padded_sequence(pad_embed, lens, batch_first=True, enforce_sorted=False)

        packed_output, (ht, ct) = self.lstm(pad_embed_pack)
        # output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        return ht[-1]


class PolicyLSTMBatch(PolicyLSTMWordBatch):

    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1, num_filters=3,
                 kernel_size=1, stride=5):
        # super(PolicyLSTMBatch, self).__init__()
        PolicyLSTMWordBatch.__init__(self, num_tokens, word_emb_size, hidden_size, num_layers=num_layers)
        self.num_filters = word_emb_size if num_filters is None else num_filters
        self.stride = stride
        self.kernel_size = kernel_size
        h_out = int((14 + 2 * 0 - 1 * (self.kernel_size - 1)) / self.stride)
        self.fc = nn.Linear(self.num_filters * h_out ** 2 + self.hidden_size,
                            num_tokens + 1)

        # self.fc = nn.Linear(self.hidden_size + self.hidden_size, num_tokens + 1)

        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=self.kernel_size,
                              stride=self.stride)
        self.img_layer = nn.Linear(1024 * 14 * 14, self.hidden_size)
        # self.pooling = pooling
        # self.max_pool = nn.MaxPool2d(kernel_size=self.pool_kernel)

    def forward(self):
        raise NotImplementedError

    def act(self, state_text, state_img, valid_actions=None):

        embed_text = self._get_embed_text(state_text)

        img_feat = state_img.to(self.device)
        img_feat_ = F.relu(self.conv(img_feat))
        img_feat__ = img_feat_.view(img_feat.size(0), -1)

        embedding = torch.cat((img_feat__, embed_text.view(embed_text.size(0), -1)), dim=1)
        out = self.fc(embedding)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]

        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        if valid_actions is not None:
            logits = torch.gather(logits, 1, valid_actions)
        probs = F.softmax(logits, dim=1)
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        return policy_dist, value



if __name__ == '__main__':
    train_features_path = '../../data/train_features.h5'
    hf = h5py.File(train_features_path, 'r')
    img_feat = hf.get('features')
    img_feat = np.array(img_feat)
    print('features shape', img_feat.shape)  # shape (num_samples, 1024, 14, 14).

    img_feat = torch.tensor(img_feat, dtype=torch.float32)
    seq_len = 10
    num_tokens = 85
    word_emb_size = 64
    hidden_size = 128
    dummy_text_input = torch.ones(img_feat.size(0), seq_len, dtype=torch.long)
    model = PolicyGRUWord(num_tokens=num_tokens, word_emb_size=word_emb_size, hidden_size=hidden_size)
    policy_dist, value = model(dummy_text_input, img_feat)
    print('policy dist', policy_dist.shape)
    print('value', value.shape)
