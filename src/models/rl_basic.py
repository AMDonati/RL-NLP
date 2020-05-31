import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_packed_sequence, pad_sequence, pack_padded_sequence


class PolicyGRU(nn.Module):
    def __init__(self, num_tokens, word_emb_size, hidden_size, num_filters=None, num_layers=1,
                 pooling=True):
        super(PolicyGRU, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if num_filters is None:
            self.num_filters = word_emb_size
        else:
            self.num_filters = num_filters
        self.pooling = pooling
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.gru = nn.GRU(word_emb_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(12 * 14 * 14 + self.hidden_size, num_tokens + 1)
        self.saved_log_probs = []
        self.rewards = []
        self.values = []
        self.last_policy = []
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=1)
        self.last_policy = []

    def forward(self, text_inputs, img_feat, valid_actions=None):
        '''
        :param text_inputs: shape (S, B)
        :param img_feat: shape (B, C, H, W)
        :param hidden: shape (num_layers, B, hidden_size)
        :return:
        log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
        '''
        embed_text = self._get_embed_text(text_inputs)

        img_feat_ = F.relu(self.conv(img_feat))
        img_feat = img_feat_.view(img_feat.size(0), -1)

        embedding = torch.cat((img_feat, embed_text.view(embed_text.size(0), -1)), dim=1)
        out = self.fc(embedding)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]

        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        if isinstance(valid_actions, dict):
            logits = logits[:, list(valid_actions.values())]
        probs = F.softmax(logits, dim=1)
        # sumprobs = probs.sum().detach().numpy()
        # if math.isnan(sumprobs):
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        self.last_policy.append(probs.detach().numpy()[0])
        return policy_dist, value

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


class PolicyGRU_Custom(nn.Module):
    def __init__(self, num_tokens, word_emb_size, hidden_size, num_filters=None, num_layers=1,
                 pooling=True):
        super(PolicyGRU_Custom, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        if num_filters is None:
            self.num_filters = word_emb_size
        else:
            self.num_filters = num_filters
        self.pooling = pooling
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.gru = nn.GRU(word_emb_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(12 * 14 * 14 + self.hidden_size, num_tokens + 1)
        self.saved_log_probs = []
        self.rewards = []
        self.values = []
        self.last_policy = []
        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=1)
        self.last_policy = []

    def forward(self):
        raise NotImplementedError

    def act(self, state):

        # text_inputs, img_feat=state.text, state.img
        # states_=[torch.cat((state_.img,state_.text.view(state_.text.size(0),-1)), dim=1) for state_ in state]
        texts = [state_.text[0] for state_ in state]

        texts.sort(key=lambda x: x.size(), reverse=True)
        text_inputs = torch.nn.utils.rnn.pack_sequence(texts)

        img_feat = torch.cat([state_.img for state_ in state])

        embed_text = self._get_embed_text(text_inputs)

        img_feat_ = F.relu(self.conv(img_feat))
        img_feat = img_feat_.view(img_feat.size(0), -1)

        embedding = torch.cat((img_feat, embed_text.view(embed_text.size(0), -1)), dim=1)
        out = self.fc(embedding)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]

        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        probs = F.softmax(logits, dim=1)
        # sumprobs = probs.sum().detach().numpy()
        # if math.isnan(sumprobs):
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        self.last_policy.append(probs.detach().numpy()[0])
        return policy_dist, value

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def update(self, text_inputs, img_feat, valid_actions=None):
        '''
        :param text_inputs: shape (S, B)
        :param img_feat: shape (B, C, H, W)
        :param hidden: shape (num_layers, B, hidden_size)
        :return:
        log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
        '''
        embed_text = self._get_embed_text(text_inputs)

        img_feat_ = F.relu(self.conv(img_feat))
        img_feat = img_feat_.view(img_feat.size(0), -1)

        embedding = torch.cat((img_feat, embed_text.view(embed_text.size(0), -1)), dim=1)
        out = self.fc(embedding)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]

        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        if isinstance(valid_actions, dict):
            logits = logits[:, list(valid_actions.values())]
        probs = F.softmax(logits, dim=1)
        # sumprobs = probs.sum().detach().numpy()
        # if math.isnan(sumprobs):
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        self.last_policy.append(probs.detach().numpy()[0])
        return policy_dist, value

    def simple_elementwise_apply(self, fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)

    def _get_embed_text(self, text):
        embs = self.simple_elementwise_apply(self.word_embedding, text)
        _, hidden = self.gru(embs)
        return hidden[-1]


class PolicyGRUWordBatch(nn.Module):

    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1):
        super(PolicyGRUWordBatch, self).__init__()
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

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        # text_inputs, img_feat=state.text, state.img
        # states_=[torch.cat((state_.img,state_.text.view(state_.text.size(0),-1)), dim=1) for state_ in state]
        texts = [state_.text[0] for state_ in state]

        # texts.sort(key=lambda x: x.size(), reverse=True)
        text_inputs = torch.nn.utils.rnn.pack_sequence(texts, enforce_sorted=False)

        # img_feat = torch.cat([state_.img for state_ in state])

        embed_text = self._get_embed_text(text_inputs)

        # img_feat_ = F.relu(self.conv(img_feat))
        # img_feat = img_feat_.view(img_feat.size(0), -1)

        # embedding = torch.cat((img_feat, embed_text.view(embed_text.size(0), -1)), dim=1)
        out = self.fc(embed_text)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]

        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        probs = F.softmax(logits, dim=1)
        # sumprobs = probs.sum().detach().numpy()
        # if math.isnan(sumprobs):
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        self.last_policy.append(probs.detach().numpy()[0])
        return policy_dist, value

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def update(self, text_inputs, img_feat, valid_actions=None):
        '''
        :param text_inputs: shape (S, B)
        :param img_feat: shape (B, C, H, W)
        :param hidden: shape (num_layers, B, hidden_size)
        :return:
        log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
        '''
        embed_text = self._get_embed_text(text_inputs)

        img_feat_ = F.relu(self.conv(img_feat))
        img_feat = img_feat_.view(img_feat.size(0), -1)

        embedding = torch.cat((img_feat, embed_text.view(embed_text.size(0), -1)), dim=1)
        out = self.fc(embedding)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]

        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        if isinstance(valid_actions, dict):
            logits = logits[:, list(valid_actions.values())]
        probs = F.softmax(logits, dim=1)
        # sumprobs = probs.sum().detach().numpy()
        # if math.isnan(sumprobs):
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        self.last_policy.append(probs.detach().numpy()[0])
        return policy_dist, value

    def _get_embed_text(self, text):
        embs = self.simple_elementwise_apply(self.word_embedding, text)
        out, hidden = self.gru(embs)
        output, input_sizes = pad_packed_sequence(out, batch_first=True)
        return hidden[-1]

    def simple_elementwise_apply(self, fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)


class PolicyLSTMWordBatch(nn.Module):

    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1):
        super(PolicyLSTMWordBatch, self).__init__()
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

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        # text_inputs, img_feat=state.text, state.img
        # states_=[torch.cat((state_.img,state_.text.view(state_.text.size(0),-1)), dim=1) for state_ in state]
        texts = [state_.text[0] for state_ in state]

        # texts.sort(key=lambda x: x.size(), reverse=True)
        #text_inputs = torch.nn.utils.rnn.pack_sequence(texts, enforce_sorted=False)

        # img_feat = torch.cat([state_.img for state_ in state])

        embed_text = self._get_embed_text(texts)

        # img_feat_ = F.relu(self.conv(img_feat))
        # img_feat = img_feat_.view(img_feat.size(0), -1)

        # embedding = torch.cat((img_feat, embed_text.view(embed_text.size(0), -1)), dim=1)
        out = self.fc(embed_text)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]

        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        probs = F.softmax(logits, dim=1)
        # sumprobs = probs.sum().detach().numpy()
        # if math.isnan(sumprobs):
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        self.last_policy.append(probs.detach().numpy()[0])
        return policy_dist, value

    def evaluate(self, state, action):
        action_probs = self.action_layer(state)
        dist = Categorical(action_probs)

        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()

        state_value = self.value_layer(state)

        return action_logprobs, torch.squeeze(state_value), dist_entropy

    def update(self, text_inputs, img_feat, valid_actions=None):
        '''
        :param text_inputs: shape (S, B)
        :param img_feat: shape (B, C, H, W)
        :param hidden: shape (num_layers, B, hidden_size)
        :return:
        log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
        '''
        embed_text = self._get_embed_text(text_inputs)

        img_feat_ = F.relu(self.conv(img_feat))
        img_feat = img_feat_.view(img_feat.size(0), -1)

        embedding = torch.cat((img_feat, embed_text.view(embed_text.size(0), -1)), dim=1)
        out = self.fc(embedding)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]

        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        if isinstance(valid_actions, dict):
            logits = logits[:, list(valid_actions.values())]
        probs = F.softmax(logits, dim=1)
        # sumprobs = probs.sum().detach().numpy()
        # if math.isnan(sumprobs):
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        self.last_policy.append(probs.detach().numpy()[0])
        return policy_dist, value

    def _get_embed_text(self, text):
        #embs = self.simple_elementwise_apply(self.word_embedding, text)
        #texts=torch.stack(text)
        padded = pad_sequence(text,batch_first=True, padding_value=0)
        lens = list(map(len, text))

        pad_embed = self.word_embedding(padded)
        pad_embed_pack = pack_padded_sequence(pad_embed, lens, batch_first=True, enforce_sorted=False)

        #lstm_inputs = torch.nn.utils.rnn.pack_sequence(pad_embed_pack, enforce_sorted=False)

        packed_output, (ht, ct) = self.lstm(pad_embed_pack)
        output, input_sizes = pad_packed_sequence(packed_output, batch_first=True)
        return ht[-1]

    def simple_elementwise_apply(self, fn, packed_sequence):
        """applies a pointwise function fn to each element in packed_sequence"""
        return torch.nn.utils.rnn.PackedSequence(fn(packed_sequence.data), packed_sequence.batch_sizes)
