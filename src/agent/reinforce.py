import torch
import torch.nn as nn
import torch.optim as optim


class REINFORCE:
    def __init__(self, model, gamma=1., lr=1e-2, pretrained_lm=None):
        self.model = model
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.pretrained_lm = pretrained_lm
        #self.max_reward = 0

    def get_top_k_words(self, state, top_k=10):
        """
        Truncate the action space with the top k words of a pretrained language model
        :param state: state
        :param top_k: number of words
        :return: top k words
        """
        if self.pretrained_lm is None:
            return None
        dist, value = self.pretrained_lm(state.text, state.img, None)
        probs = dist.probs
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
        valid_actions = {i: token for i, token in enumerate(top_k_indices.numpy()[0])}
        return valid_actions

    def select_action(self, state, forced=None, num_truncated=10):
        valid_actions = self.get_top_k_words(state, num_truncated)
        m, value = self.model(state.text, state.img, valid_actions)
        action = m.sample() if forced is None else forced
        log_prob = m.log_prob(action).view(1)
        if isinstance(valid_actions, dict):
            action = torch.tensor(valid_actions[action.item()]).view(1)
        return action.item(), log_prob, value, valid_actions, m

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        mse = nn.MSELoss()
        #if self.max_reward<sum(self.model.rewards) :
        #    self.max_reward = sum(self.model.rewards)
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()
        for log_prob, R, value in zip(self.model.saved_log_probs, returns, self.model.values):
            policy_loss.append(-log_prob * (R - value))
            ms = mse(value, R)
            #ms = mse(value, torch.tensor(self.max_reward).float())
            policy_loss.append(ms.view(1))
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.model.rewards[:]
        del self.model.saved_log_probs[:]
        del self.model.values[:]
