import torch
import torch.nn as nn

from agent.agent import Agent


class REINFORCE(Agent):
    def __init__(self, policy, env, gamma=1., lr=1e-2, pretrained_lm=None, update_episode=1, word_emb_size=8,
                 hidden_size=24):
        Agent.__init__(self, policy, env, gamma=gamma, lr=lr, pretrained_lm=pretrained_lm, word_emb_size=word_emb_size,
                       hidden_size=hidden_size)
        self.update_episode = update_episode
        self.MSE_loss = nn.MSELoss()
        self.update_mode = "episode"

    def select_action(self, state, num_truncated=10, forced=None):
        valid_actions = self.get_top_k_words([state], num_truncated)
        m, value = self.policy.act([state], valid_actions)
        action = m.sample() if forced is None else forced
        log_prob = m.log_prob(action).view(-1)
        self.memory.actions.append(action)
        if valid_actions is not None:
            action = torch.gather(valid_actions, 1, action.view(1, 1))
        self.memory.states.append(state)
        self.memory.logprobs.append(log_prob)
        self.memory.values.append(value)
        return action.numpy(), log_prob, value, None, m

    def update(self):

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - is_terminal))
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).to(self.device).float().view(-1, 1)
        logprobs = torch.stack(self.memory.logprobs).to(self.device)
        values = torch.stack(self.memory.values)

        advantages = rewards - values
        loss = -logprobs * advantages + self.MSE_loss(values, rewards)
        # take gradient step
        self.optimizer.zero_grad()
        loss.mean().backward()
        self.optimizer.step()
