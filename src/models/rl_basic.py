import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical


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
        self.last_policy.append(probs.detach().numpy()[0])
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

    def forward(self, text_inputs, img_feat, valid_actions=None):
        embed_text = self._get_embed_text(text_inputs)
        out = self.fc(embed_text)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        if isinstance(valid_actions, dict):
            logits = logits[:, list(valid_actions.values())]
        probs = F.softmax(logits, dim=1)
        policy_dist = Categorical(probs)
        probs_ = policy_dist.probs.clone()
        self.last_policy.append(probs_.detach().numpy()[0])
        return policy_dist, value

    def _get_embed_text(self, text):
        _, hidden = self.gru(self.word_embedding(text))
        return hidden[-1]
