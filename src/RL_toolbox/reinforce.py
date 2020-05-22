import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class REINFORCE:
    def __init__(self, env, model, optimizer, device, gamma=1, mode='sampling', debug=False):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.mode = mode
        self.env = env
        self.debug = debug
        if debug:
            print('reducing action space for debugging...')
        self.action_space = None

    def select_action(self, state):
        m, _, value = self.model(state.text, state.img)
        action = m.sample()
        return action.item(), m.log_prob(action).view(1), value.view(1)

    def generate_one_episode(self):
        state, ep_reward = self.env.reset(), 0
        for t in range(0, self.env.max_len + 1):
            action, log_probs, value = self.select_action(state)
            state, (reward, closest_question), done, _ = self.env.step(action)
            self.model.rewards.append(reward)
            self.model.values.append(value)
            self.model.saved_log_probs.append(log_probs)
            ep_reward += reward
            if done:
                break

        episode = self.env.Episode(self.env.img_idx, self.env.img_feats.data.numpy(), self.env.ref_questions_decoded,
                                   closest_question,
                                   self.env.dialog, self.model.rewards)
        returns = self.compute_returns(self.model.rewards)
        return self.model.saved_log_probs, returns, self.model.values, episode


    def train_batch(self, returns, log_probs, values):
        mse = nn.MSELoss(reduction='none')
        policy_loss = -log_probs * (returns - values) # (B,S)
        value_loss = mse(values, returns) # (B,S)
        loss = policy_loss + value_loss
        loss = loss.sum(-1).mean(0)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        del self.model.rewards[:]
        del self.model.saved_log_probs[:]
        del self.model.values[:]
        return loss

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        mse = nn.MSELoss()
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()

        for log_prob, R, value in zip(self.model.saved_log_probs, returns, self.model.values):
            policy_loss.append(-log_prob * (R - value))
            ms = mse(value, R)
            policy_loss.append(ms.view(1))

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()

        del self.model.rewards[:]
        del self.model.saved_log_probs[:]
        del self.model.values[:]

        return policy_loss

    def compute_returns(self, rewards):
        R = 0
        returns = []
        for r in rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        return returns

    # def generate_one_episode_old(self):
    #     state = self.env.reset()
    #     if self.debug:
    #         self.action_space, _ = self.env.get_reduced_action_space()
    #     done = False
    #     rewards, log_probs, values = [], [], []
    #     while not done:
    #         action, log_prob, value = self.select_action(state)
    #         # compute next state, done, reward from the action.
    #         state, (reward, closest_question), done, _ = self.env.step(action)
    #         rewards.append(reward)
    #         log_probs.append(log_prob)
    #         values.append(value)
    #
    #     episode = self.env.Episode(self.env.img_idx, self.env.img_feats.data.numpy(), self.env.ref_questions_decoded,
    #                                closest_question,
    #                                self.env.dialog, rewards)
    #     if len(rewards) < self.env.max_len:
    #         assert state.text[:, -1] == self.env.special_tokens.EOS_idx
    #     returns = self.compute_returns(rewards)
    #
    #     return log_probs, returns, values, episode

    # def select_action_old(self, state):
    #     bs, seq_len = state.text.size(0), state.text.size(1)
    #     self.model.train()
    #     state.text.to(self.device)
    #     state.img.to(self.device)
    #     logits, _, values = self.model(state.text, state.img)  # logits > shape (s*num_samples, num_tokens)
    #     if self.action_space is not None:
    #         list_actions = list(self.action_space.values())
    #         logits = logits[:, list_actions]
    #     logits = logits.view(bs, seq_len, -1)
    #     values = values.view(bs, seq_len, -1)
    #     probas = F.softmax(logits, dim=-1)  # (num samples, s, num_tokens)
    #     if self.mode == 'sampling':
    #         m = Categorical(probas[:, -1, :]) # multinomial distribution with weights = probas.
    #         action = m.sample()
    #     elif self.mode == 'greedy':
    #         _, action = probas[:, -1, :].max(dim=-1)
    #         action = action.squeeze(-1)
    #     log_prob = F.log_softmax(logits, dim=-1)[:, -1, action]
    #     if self.action_space is not None:
    #         action = self.action_space[action.item()]
    #     return action, log_prob, values[:, -1, :]
