import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from RL_toolbox.RL_functions import masked_softmax


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

    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1,
                 rl=True, truncate_mode="masked", **kwargs):
        super(PolicyLSTMWordBatch, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.lstm = nn.LSTM(word_emb_size, self.hidden_size, batch_first=True)
        self.rl = rl
        self.action_head = nn.Linear(self.hidden_size,
                                     num_tokens)
        self.value_head = nn.Linear(self.hidden_size, 1)

        truncature = {"masked": self.mask_truncature, "gather": self.gather_truncature,
                      "masked_inf": self.mask_inf_truncature}
        #self.truncate = truncature[truncate_mode]
        self.truncate = truncature["masked"]

    def forward(self, state_text, state_img, valid_actions=None):
        embedding = self._get_embed_text(state_text)
        logits = self.action_head(embedding)  # (B,S,num_tokens)
        value = self.value_head(embedding)
        probs = F.softmax(logits, dim=-1)
        policy_dist = Categorical(probs)
        if valid_actions is not None:
            policy_dist_truncated = self.truncate(valid_actions, logits)
        else:
            policy_dist_truncated = policy_dist
        return policy_dist, policy_dist_truncated, value

    def _get_embed_text(self, text):
        # padded = pad_sequence(text, batch_first=True, padding_value=0).to(self.device)
        lens = (text != 0).sum(dim=1)
        pad_embed = self.word_embedding(text.to(self.device))
        pad_embed_pack = pack_padded_sequence(pad_embed, lens, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(pad_embed_pack)
        return ht[-1]

    def gather_truncature(self, valid_actions, logits):
        logits = torch.gather(logits.clone().detach(), -1, valid_actions)
        probs = F.softmax(logits, dim=-1)
        return Categorical(probs)

    def mask_truncature(self, valid_actions, logits):
        mask = torch.zeros(logits.size(0), self.num_tokens).to(self.device)
        mask[:, valid_actions] = 1
        probs_truncated = masked_softmax(logits.clone().detach(), mask)
        # check that the truncation is right.
        sum_probs_va = torch.round(probs_truncated[:, valid_actions].sum(dim=-1))
        assert torch.all(sum_probs_va - torch.ones(sum_probs_va.size()) < 1e-6), "ERROR IN TRUNCATION FUNCTION"
        #if not torch.all(torch.eq(sum_probs_va, torch.ones(sum_probs_va.size()))):
            #print(sum_probs_va)
        policy_dist_truncated = Categorical(probs_truncated)
        return policy_dist_truncated

    def mask_inf_truncature(self, valid_actions, logits):
        mask = torch.ones(logits.size(0), self.num_tokens) * -1e32
        mask[:, valid_actions] = logits[:, valid_actions].clone().detach()
        probs_truncated = F.softmax(mask, dim=-1)
        # check that the truncation is right.
        assert probs_truncated[:, valid_actions].sum(dim=-1) == 1, "ERROR IN TRUNCATION FUNCTION"
        policy_dist_truncated = Categorical(probs_truncated)
        return policy_dist_truncated


class PolicyLSTMBatch(PolicyLSTMWordBatch):

    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1, num_filters=3,
                 kernel_size=1, stride=5, rl=True, truncate_mode="masked"):
        PolicyLSTMWordBatch.__init__(self, num_tokens, word_emb_size, hidden_size, num_layers=num_layers,
                                     truncate_mode=truncate_mode)
        self.num_filters = word_emb_size if num_filters is None else num_filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.rl = rl
        h_out = int((14 + 2 * 0 - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
        self.action_head = nn.Linear(self.num_filters * h_out ** 2 + self.hidden_size,
                                     num_tokens)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=self.kernel_size,
                              stride=self.stride)
        self.value_head = nn.Linear(self.num_filters * h_out ** 2 + self.hidden_size, 1)

    def forward(self, state_text, state_img, valid_actions=None):
        embed_text = self._get_embed_text(state_text)
        img_feat = state_img.to(self.device)
        img_feat_ = F.relu(self.conv(img_feat))
        img_feat__ = img_feat_.view(img_feat.size(0), -1)
        embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).
        logits = self.action_head(embedding)  # (B,S,num_tokens)
        value = self.value_head(embedding)
        probs = F.softmax(logits, dim=-1)
        policy_dist = Categorical(probs)
        if valid_actions is not None:
            policy_dist_truncated = self.truncate(valid_actions, logits)
        else:
            policy_dist_truncated = policy_dist
        return policy_dist, policy_dist_truncated, value


class PolicyLSTMWordBatch_SL(nn.Module):
    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1, kernel_size=1, stride=5, num_filters=3):
        super(PolicyLSTMWordBatch_SL, self).__init__()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.lstm = nn.LSTM(word_emb_size, self.hidden_size, batch_first=True)
        self.action_head = nn.Linear(self.hidden_size,
                                     num_tokens)

    def forward(self, state_text, state_img):
        embedding = self._get_embed_text(state_text)
        logits = self.action_head(embedding)  # (B,S,num_tokens)
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        value = None
        return logits, value

    def _get_embed_text(self, text):
        # padded = pad_sequence(text, batch_first=True, padding_value=0).to(self.device)
        lens = (text != 0).sum(dim=1)
        pad_embed = self.word_embedding(text)
        pad_embed_pack = pack_padded_sequence(pad_embed, lens, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(pad_embed_pack)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True, total_length=text.size(1))
        return output


class PolicyLSTMBatch_SL(PolicyLSTMWordBatch_SL):

    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1, num_filters=3,
                 kernel_size=1, stride=5):
        PolicyLSTMWordBatch_SL.__init__(self, num_tokens, word_emb_size, hidden_size, num_layers=num_layers)
        self.num_filters = word_emb_size if num_filters is None else num_filters
        self.stride = stride
        self.kernel_size = kernel_size
        h_out = int((14 + 2 * 0 - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
        self.action_head = nn.Linear(self.num_filters * h_out ** 2 + self.hidden_size,
                                     num_tokens)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=self.kernel_size,
                              stride=self.stride)

    def forward(self, state_text, state_img):
        embed_text = self._get_embed_text(state_text)
        seq_len = embed_text.size(1)
        img_feat = state_img.to(self.device)
        img_feat_ = F.relu(self.conv(img_feat))
        img_feat__ = img_feat_.view(img_feat.size(0), -1).unsqueeze(1).repeat(1, seq_len,
                                                                              1)  # repeat img along the sequence axis.
        embedding = torch.cat((img_feat__, embed_text), dim=-1)
        logits = self.action_head(embedding)  # (B,S,num_tokens)
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        value = None
        return logits, value


if __name__ == '__main__':
    train_features_path = '../../data/train_features.h5'
    hf = h5py.File(train_features_path, 'r')
    img_feat = hf.get('features')
    img_feat = np.array(img_feat)
    print('features shape', img_feat.shape)  # shape (num_samples, 1024, 14, 14).

    img_feat = torch.tensor(img_feat, dtype=torch.float32)
    img_feat_RL = img_feat[0].unsqueeze(0)
    seq_len = 10
    num_tokens = 85
    word_emb_size = 64
    hidden_size = 128
    dummy_text_input = torch.ones(img_feat.size(0), seq_len, dtype=torch.long)
    dummy_text_input_RL = torch.ones(img_feat_RL.size(0), seq_len, dtype=torch.long)

    # RL mode.
    model = PolicyLSTMBatch(num_tokens=num_tokens, word_emb_size=word_emb_size, hidden_size=hidden_size)
    policy_dist, policy_dist_truncated, value = model(dummy_text_input_RL, img_feat_RL, valid_actions=[0,4,8,10])
    model = PolicyLSTMBatch(num_tokens=num_tokens, word_emb_size=word_emb_size, hidden_size=hidden_size, truncate_mode="masked_inf")
    policy_dist, policy_dist_truncated, value = model(dummy_text_input_RL, img_feat_RL, valid_actions=[0, 4, 8, 10])

    # SL mode.
    model = PolicyLSTMBatch_SL(num_tokens=num_tokens, word_emb_size=word_emb_size, hidden_size=hidden_size)
    logits, value = model(dummy_text_input, img_feat)
    print(logits.shape)
