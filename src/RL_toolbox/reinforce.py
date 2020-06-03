import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

class REINFORCE:
    def __init__(self, env, model, optimizer, device, pretrained_lm_path=None, gamma=1, mode='sampling', debug=False):
        self.model = model
        self.optimizer = optimizer
        self.gamma = gamma
        self.device = device
        self.mode = mode
        self.env = env
        self.debug = debug
        if debug:
            print('reducing action space for debugging...')
        if pretrained_lm_path is None:
            self.pretrained_lm = None
        else:
            with open(pretrained_lm_path, 'rb') as f:
                self.pretrained_lm = torch.load(f, map_location=device)

    def get_top_k_words(self, state, k=10):
        seq_len = state.text.size(1)
        if self.pretrained_lm is None:
            return None
        log_probas, _ = self.pretrained_lm(state.text)
        log_probas = log_probas.view(state.text.size(0), seq_len, -1)
        log_probas = log_probas[:,-1,:]
        top_k_weights, top_k_indices = torch.topk(log_probas, k, sorted=True)
        valid_actions = {i: token for i, token in enumerate(top_k_indices.numpy()[0])}
        return valid_actions

    def select_action(self, state, valid_actions=None):
        text = state.text.to(self.device)
        img = state.img.to(self.device)
        m, _, value = self.model(text, img, valid_actions)
        if self.mode == 'sampling':
            action = m.sample()
        elif self.mode == 'greedy':
            action = m.probs.argmax() #TODO: debug greedy.
        if isinstance(valid_actions, dict):
            action_idx = torch.tensor(valid_actions[action.item()]).view(1)
        else:
            action_idx = action
        return action_idx.item(), m.log_prob(action).view(1), value.view(1)


    def generate_one_episode(self):
        state, ep_reward = self.env.reset(), 0
        list_valid_actions = []
        for t in range(0, self.env.max_len + 1):
            valid_actions = self.get_top_k_words(state)
            if self.pretrained_lm is not None:
                list_valid_actions.append(list(valid_actions.values()))
            action, log_probs, value = self.select_action(state, valid_actions)
            state, (reward, closest_question), done, _ = self.env.step(action)
            self.model.rewards.append(reward)
            self.model.values.append(value)
            self.model.saved_log_probs.append(log_probs)
            ep_reward += reward
            if done:
                break

        episode = self.env.Episode(self.env.img_idx,
                                   closest_question,
                                   self.env.dialog, self.model.rewards, list_valid_actions)
        self.env.current_episode = episode
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
        mse = nn.MSELoss().to(self.device)
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()

        for log_prob, R, value in zip(self.model.saved_log_probs, returns, self.model.values):
           #R = R.to(self.device) #TODO: R needs to be to.self(device) but generates bug...
            policy_loss.append(-log_prob * (R - value))
            ms = mse(value, R).to(self.device)
            policy_loss.append(ms.view(1))

        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        #policy_loss = policy_loss.to(self.device)
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
