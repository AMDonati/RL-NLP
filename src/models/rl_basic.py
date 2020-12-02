import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torchcontrib import nn as contrib_nn

from RL_toolbox.truncation import mask_truncature, mask_inf_truncature


class PolicyLSTMBatch(nn.Module):

    def __init__(self, num_tokens, word_emb_size, hidden_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_layers=1, num_filters=3,
                 kernel_size=1, stride=5, fusion="cat", env=None,
                 condition_answer="none"):
        super(PolicyLSTMBatch, self).__init__()
        self.device = device
        self.condition_answer = condition_answer
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.lstm = nn.LSTM(word_emb_size, self.hidden_size, batch_first=True)
        truncature = {"masked": mask_truncature, "masked_inf": mask_inf_truncature}
        self.truncate = truncature["masked_inf"]
        self.answer_embedding = nn.Embedding(env.dataset.len_vocab_answer, word_emb_size)
        self.fusion = fusion
        self.num_filters = word_emb_size if num_filters is None else num_filters
        self.stride = stride
        self.kernel_size = kernel_size
        h_out = int((14 + 2 * 0 - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=self.kernel_size,
                              stride=self.stride)
        if self.fusion == "none":
            self.fusion_dim = self.hidden_size
        elif self.fusion == "film":
            self.gammabeta = nn.Linear(self.hidden_size, 2 * self.num_filters)
            self.film = contrib_nn.FiLM()
            self.fusion_dim = self.num_filters * h_out ** 2
        elif self.fusion == "average":
            self.projection = nn.Linear(2048, hidden_size)
            self.avg_pooling = nn.AvgPool1d(kernel_size=101)
            self.fusion_dim = 2 * hidden_size
        else:
            self.fusion_dim = self.num_filters * h_out ** 2 + self.hidden_size

        if self.condition_answer == "after_fusion":
            self.fusion_dim += word_emb_size

        self.action_head = nn.Linear(self.fusion_dim, num_tokens)
        self.value_head = nn.Linear(self.fusion_dim, 1)

    def forward(self, state_text, state_img, state_answer=None, valid_actions=None, logits_lm=0, alpha=0.):
        embed_text = self._get_embed_text(state_text, state_answer)
        state_answer = state_answer if state_answer is None else state_answer.to(self.device)
        img_feat = state_img.to(self.device) # shape (1, 1024, 14, 14) vs (1,101,2048)
        img_feat_ = img_feat if self.fusion == "average" or self.fusion == "none" else F.relu(self.conv(img_feat)) # shape (1,3,7,7)
        embedding = self.process_fusion(embed_text, img_feat_, img_feat, state_answer)
        logits = self.action_head(embedding)  # (B,S,num_tokens)
        value = self.value_head(embedding)
        # adding lm logits bonus
        logits_exploration = (1 - alpha) * logits + alpha * logits_lm
        policy_dist, policy_dist_truncated = self.get_policies(valid_actions, logits_exploration)
        return policy_dist, policy_dist_truncated, value

    def get_policies(self, valid_actions, logits_exploration):
        probs = F.softmax(logits_exploration, dim=-1)
        policy_dist = Categorical(probs)
        if valid_actions is not None:
            policy_dist_truncated = self.truncate(valid_actions, logits_exploration, device=self.device, num_tokens=self.num_tokens)
        else:
            policy_dist_truncated = Categorical(F.softmax(logits_exploration, dim=-1))
        if torch.isnan(policy_dist_truncated.probs).any():
            print("policy dist truncated with nan")
        return policy_dist, policy_dist_truncated

    def process_fusion(self, embed_text, img_feat_, img_feat, answer):
        if self.fusion == "none":
            embedding = embed_text
        elif self.fusion == "film":
            gammabeta = self.gammabeta(embed_text).view(-1, 2, self.num_filters)
            gamma, beta = gammabeta[:, 0, :], gammabeta[:, 1, :]
            embedding = self.film(img_feat_, gamma, beta).view(img_feat.size(0), -1)
        elif self.fusion == "average":
            img_feat__ = self.projection(img_feat_) #(1,101,64)
            img_feat__ = img_feat__.transpose(2,1)
            img_feat__ = self.avg_pooling(img_feat__) #(1,64,1)
            img_feat__ = img_feat__.squeeze(dim=-1)
            embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).
        else:
            img_feat__ = img_feat_.view(img_feat.size(0), -1)
            embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).
        if self.condition_answer == "after_fusion" and answer is not None:
            embedding = torch.cat([embedding, self.answer_embedding(answer.view(-1))], dim=1)
        return embedding

    def _get_embed_text(self, text, answer):
        lens = (text != 0).sum(dim=1)
        pad_embed = self.word_embedding(text.to(self.device))
        if self.condition_answer == "before_lstm" and answer is not None:
            pad_embed = torch.cat([pad_embed, self.answer_embedding(answer.view(text.size(0), 1)).to(self.device)],
                                  dim=1)
        pad_embed_pack = pack_padded_sequence(pad_embed, lens, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(pad_embed_pack)
        return ht[-1]


class PolicyLSTMBatch_SL(nn.Module):
    def __init__(self, num_tokens, word_emb_size, hidden_size, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), num_layers=1, num_filters=3,
                 kernel_size=1, stride=5, fusion="cat", condition_answer="none", num_tokens_answer=32):
        super(PolicyLSTMBatch_SL, self).__init__()

        self.device = device
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_filters = word_emb_size if num_filters is None else num_filters
        self.stride = stride
        self.kernel_size = kernel_size
        self.fusion = fusion
        self.condition_answer = condition_answer

        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.lstm = nn.LSTM(word_emb_size, self.hidden_size, batch_first=True)
        self.answer_embedding = nn.Embedding(num_tokens_answer, word_emb_size)
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=self.kernel_size,
                              stride=self.stride)

        h_out = int((14 + 2 * 0 - 1 * (self.kernel_size - 1) - 1) / self.stride + 1)

        if self.fusion == "film":
            self.gammabeta = nn.Linear(self.hidden_size, 2 * self.num_filters)
            self.film = contrib_nn.FiLM()
            self.fusion_dim = self.num_filters * h_out ** 2
        elif self.fusion == "average":
            self.projection = nn.Linear(2048, hidden_size)
            self.avg_pooling = nn.AvgPool1d(kernel_size=101)
            self.fusion_dim = 2 * hidden_size
        else:
            self.fusion_dim = self.num_filters * h_out ** 2 + self.hidden_size

        if self.condition_answer == "after_fusion":
            self.fusion_dim += word_emb_size

        self.action_head = nn.Linear(self.fusion_dim, num_tokens)

    def _get_embed_text(self, text):
        lens = (text != 0).sum(dim=1)
        pad_embed = self.word_embedding(text)
        pad_embed_pack = pack_padded_sequence(pad_embed, lens, batch_first=True, enforce_sorted=False)
        packed_output, (ht, ct) = self.lstm(pad_embed_pack)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True, total_length=text.size(1))
        return output

    def process_fusion(self, embed_text, img_feat_, img_feat, answer, seq_len):
        if self.fusion == "film":
            gammabeta = self.gammabeta(embed_text).view(embed_text.size(0), embed_text.size(1), 2, self.num_filters)
            gamma, beta = gammabeta[:, :, 0, :], gammabeta[:, :, 1, :]
            img_feat__ = img_feat_.unsqueeze(1).repeat(1, seq_len, 1, 1, 1).view(-1, img_feat_.size(1),
                                                                                 img_feat_.size(2),
                                                                                 img_feat_.size(2))
            embedding = self.film(img_feat__, gamma.view(-1, gamma.size(2)), beta.view(-1, beta.size(2)))
            embedding = embedding.view(embed_text.size(0), embed_text.size(1), -1)
        elif self.fusion == "average":
            img_feat__ = img_feat_.transpose(2, 1) # (B, 2048, 101)
            img_feat__ = self.avg_pooling(img_feat__).squeeze(-1) # (B,2048)
            img_feat__ = self.projection(img_feat__)  # (B,hidden_size)
            img_feat__ = img_feat__.unsqueeze(1).repeat(1,seq_len,1)
            embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).
        else:
            img_feat__ = img_feat_.view(img_feat.size(0), -1).unsqueeze(1).repeat(1, seq_len, 1)
            embedding = torch.cat((img_feat__, embed_text), dim=-1)  # (B,S,hidden_size).

        if self.condition_answer == "after_fusion" and answer is not None:
            repeated_answer = self.answer_embedding(answer).unsqueeze(1).repeat(1, seq_len, 1)
            embedding = torch.cat([embedding, repeated_answer], dim=2)

        return embedding

    def forward(self, state_text, state_img, state_answer):
        embed_text = self._get_embed_text(state_text)
        seq_len = embed_text.size(1)
        img_feat = state_img.to(self.device)
        img_feat_ = img_feat if self.fusion == "average" or self.fusion == "none" else F.relu(self.conv(img_feat))
        embedding = self.process_fusion(embed_text=embed_text, img_feat_=img_feat_, img_feat=img_feat, answer=state_answer, seq_len=seq_len)
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
