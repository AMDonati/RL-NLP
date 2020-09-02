import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcontrib import nn as contrib_nn

from RL_toolbox.truncation import gather_truncature, mask_truncature, mask_inf_truncature


class PolicyLSTMBatch(nn.Module):

    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1, num_filters=3,
                 kernel_size=1, stride=5, train_policy="all_space", fusion="cat", env=None,
                 condition_answer="none"):
        super(PolicyLSTMBatch, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.condition_answer = condition_answer
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.lstm = nn.LSTM(word_emb_size, self.hidden_size, batch_first=True)
        truncature = {"masked": mask_truncature, "gather": gather_truncature, "masked_inf": mask_inf_truncature}
        self.truncate = truncature["masked"]
        self.train_policy = train_policy
        self.answer_embedding = nn.Embedding(env.clevr_dataset.len_vocab_answer, word_emb_size)
        # self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.fusion = fusion
        self.num_filters = word_emb_size if num_filters is None else num_filters
        self.stride = stride
        self.kernel_size = kernel_size
        h_out = int((14 + 2 * 0 - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=self.kernel_size,
                              stride=self.stride)
        if self.fusion == "film":
            self.gammabeta = nn.Linear(self.hidden_size, 2 * self.num_filters)
            self.film = contrib_nn.FiLM()
            self.fusion_dim = self.num_filters * h_out ** 2
        else:
            self.fusion_dim = self.num_filters * h_out ** 2 + self.hidden_size

        if self.condition_answer == "after_fusion":
            self.fusion_dim += word_emb_size

        self.action_head = nn.Linear(self.fusion_dim, num_tokens)
        self.value_head = nn.Linear(self.fusion_dim, 1)

    def forward(self, state_text, state_img, state_answer=None, valid_actions=None, logits_lm=0, alpha=0.):
        embed_text = self._get_embed_text(state_text, state_answer)
        img_feat = state_img.to(self.device)
        img_feat_ = F.relu(self.conv(img_feat))
        embedding = self.process_fusion(embed_text, img_feat_, img_feat, state_answer)
        logits = self.action_head(embedding)  # (B,S,num_tokens)
        value = self.value_head(embedding)
        # adding lm logits bonus
        logits_exploration = (1 - alpha) * logits + alpha * logits_lm
        probs = F.softmax(logits, dim=-1)
        policy_dist, policy_dist_truncated = self.get_policies(probs, valid_actions, logits_exploration)
        return policy_dist, policy_dist_truncated, value

    def get_policies(self, probs, valid_actions, logits_exploration):
        policy_dist = Categorical(probs)
        if valid_actions is not None:
            policy_dist_truncated = self.truncate(valid_actions, logits_exploration, self.num_tokens)
            if self.train_policy == 'truncated':
                policy_dist = policy_dist_truncated
        else:
            policy_dist_truncated = Categorical(F.softmax(logits_exploration, dim=-1))
        return policy_dist, policy_dist_truncated

    def process_fusion(self, embed_text, img_feat_, img_feat, answer):
        if self.fusion == "film":
            gammabeta = self.gammabeta(embed_text).view(-1, 2, self.num_filters)
            gamma, beta = gammabeta[:, 0, :], gammabeta[:, 1, :]
            embedding = self.film(img_feat_, gamma, beta).view(img_feat.size(0), -1)
        else:
            img_feat__ = img_feat_.view(img_feat.size(0), -1)
            embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).
        if self.condition_answer == "after_fusion" and answer is not None:
            embedding = torch.cat([embedding, self.answer_embedding(answer.view(-1))], dim=1)
        return embedding

    def _get_embed_text(self, text, answer):
        # padded = pad_sequence(text, batch_first=True, padding_value=0).to(self.device)
        lens = (text != 0).sum(dim=1)
        pad_embed = self.word_embedding(text.to(self.device))
        if self.condition_answer == "before_lstm" and answer is not None:
            pad_embed = torch.cat([pad_embed, self.answer_embedding(answer.view(text.size(0), 1))], dim=1)
            # text = torch.cat([answer.view(text.size(0), 1), text], dim=1)

        pad_embed_pack = pack_padded_sequence(pad_embed, lens, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(pad_embed_pack)
        return ht[-1]


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
                 kernel_size=1, stride=5, fusion="cat"):
        PolicyLSTMWordBatch_SL.__init__(self, num_tokens, word_emb_size, hidden_size, num_layers=num_layers)
        self.num_filters = word_emb_size if num_filters is None else num_filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.fusion = fusion
        h_out = int((14 + 2 * 0 - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
        if self.fusion == "film":
            self.gammabeta = nn.Linear(self.hidden_size, 2 * self.num_filters)
            self.film = contrib_nn.FiLM()
            self.fusion_dim = self.num_filters * h_out ** 2
        else:
            self.fusion_dim = self.num_filters * h_out ** 2 + self.hidden_size

        self.action_head = nn.Linear(self.fusion_dim, num_tokens)

        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=self.kernel_size,
                              stride=self.stride)

    def forward(self, state_text, state_img):
        embed_text = self._get_embed_text(state_text)
        seq_len = embed_text.size(1)
        img_feat = state_img.to(self.device)
        img_feat_ = F.relu(self.conv(img_feat))

        # img_feat__ = img_feat_.view(img_feat.size(0), -1).unsqueeze(1).repeat(1, seq_len, 1)
        # embedding = torch.cat((img_feat__, embed_text), dim=-1)
        if self.fusion == "film":
            gammabeta = self.gammabeta(embed_text).view(embed_text.size(0), embed_text.size(1), 2, self.num_filters)
            gamma, beta = gammabeta[:, :, 0, :], gammabeta[:, :, 1, :]
            img_feat__ = img_feat_.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).view(-1, img_feat_.size(1),
                                                                                 img_feat_.size(2),
                                                                                 img_feat_.size(2))
            embedding = self.film(img_feat__, gamma.view(-1, gamma.size(2)), beta.view(-1, beta.size(2)))
            embedding = embedding.view(embed_text.size(0), embed_text.size(1), -1)
        else:
            img_feat__ = img_feat_.view(img_feat.size(0), -1).unsqueeze(1).repeat(1, seq_len, 1)
            embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).

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
    policy_dist, policy_dist_truncated, value = model(dummy_text_input_RL, img_feat_RL, valid_actions=[0, 4, 8, 10])
    model = PolicyLSTMBatch(num_tokens=num_tokens, word_emb_size=word_emb_size, hidden_size=hidden_size,
                            truncate_mode="masked_inf")
    policy_dist, policy_dist_truncated, value = model(dummy_text_input_RL, img_feat_RL, valid_actions=[0, 4, 8, 10])

    # SL mode.
    model = PolicyLSTMBatch_SL(num_tokens=num_tokens, word_emb_size=word_emb_size, hidden_size=hidden_size)
    logits, value = model(dummy_text_input, img_feat)
    print(logits.shape)
