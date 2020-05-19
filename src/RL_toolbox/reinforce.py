import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import namedtuple


class REINFORCE:
    def __init__(self, env, model, optimizer, device, gamma=1, mode='sampling'):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.mode = mode
        self.env = env

    def select_action(self, state):
        # TODO, here: add an action space.
        bs, seq_len = state.text.size(0), state.text.size(1)
        self.model.train()
        state.text.to(self.device)
        state.img.to(self.device)
        logits, hidden, values = self.model(state.text, state.img)  # logits > shape (s*num_samples, num_tokens)
        logits = logits.view(bs, seq_len, -1)  # TODO: prune action space.
        values = values.view(bs, seq_len, -1)
        probas = F.softmax(logits, dim=-1)  # (num samples, s, num_tokens)
        if self.mode == 'sampling':
            m = Categorical(probas[:, -1, :])  # multinomial distribution with weights = probas.
            action = m.sample()
        elif self.mode == 'greedy':
            _, action = probas[:, -1, :].max(dim=-1)
            action = action.squeeze(-1)
        log_prob = F.log_softmax(logits, dim=-1)[:, -1, action]
        return action, log_prob, values[:, -1, :]

    def generate_one_episode(self):
        state = self.env.reset()
        done = False
        rewards, log_probs, values = [], [], []
        while not done:
            action, log_prob, value = self.select_action(state)
            # compute next state, done, reward from the action.
            state, (reward, closest_question), done, _ = self.env.step(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            values.append(value)

        episode = self.env.Episode(self.env.img_idx, self.env.img_feats.data.numpy(), self.env.ref_questions_decoded,
                                   closest_question,
                                   self.env.dialog, rewards)
        if len(rewards) < self.env.max_len:
            assert state.text[:, -1] == self.env.special_tokens.EOS_idx
        returns = self.compute_returns(rewards)

        return log_probs, returns, values, episode

    def train_batch(self, returns, log_probs, values):
        mse = nn.MSELoss(reduction='none')
        policy_loss = -log_probs * (returns - values) # (B,S)
        value_loss = mse(values, returns) # (B,S)
        loss = policy_loss + value_loss
        loss = loss.sum(-1).mean(0)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        return loss

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns
