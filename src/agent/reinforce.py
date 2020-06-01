import torch
import torch.nn as nn

from agent.agent import Agent


class REINFORCE(Agent):
    def __init__(self, model, env, gamma=1., lr=1e-2, pretrained_lm=None):
        Agent.__init__(self, model, env, gamma=gamma, lr=lr, pretrained_lm=pretrained_lm)
        self.update_timestep = self.env.max_len
        self.MSE_loss = nn.MSELoss()

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
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
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
