import logging

import numpy as np
import torch
import torch.nn as nn

from agent.agent import Agent


class REINFORCE(Agent):
    def __init__(self, policy, env, writer, gamma=1., lr=1e-2, pretrained_lm=None, lm_sl=True, pretrained_policy=None,
                 pretrain=False, update_every=50, word_emb_size=8, hidden_size=24, kernel_size=1, stride=2,
                 num_filters=3, num_truncated=10, truncate_mode="masked"):

        Agent.__init__(self, policy, env, writer=writer, gamma=gamma, lr=lr, pretrained_lm=pretrained_lm,
                       lm_sl=lm_sl,
                       pretrained_policy=pretrained_policy, pretrain=pretrain, update_every=update_every,
                       word_emb_size=word_emb_size, hidden_size=hidden_size, kernel_size=kernel_size, stride=stride,
                       num_filters=num_filters, num_truncated=num_truncated, truncate_mode=truncate_mode)
        self.update_every = update_every
        self.MSE_loss = nn.MSELoss(reduction="none")
        self.update_mode = "episode"
        self.writer_iteration = 0

    def select_action(self, state, num_truncated=10, forced=None):
        valid_actions, actions_probs = self.get_top_k_words(state.text, num_truncated)
        m, policy_dist_truncated, value = self.policy(state.text, state.img, valid_actions)
        action = m.sample() if forced is None else forced
        log_prob = m.log_prob(action.to(self.device)).view(-1)

        return action.cpu().numpy(), log_prob, value, (valid_actions, actions_probs), m

    def update(self):
        # getting the lengths for the loss split
        is_terminals = np.array(self.memory.is_terminals)
        lengths = np.argwhere(is_terminals == True) + 1
        lengths[1:] = lengths[1:] - lengths[:-1]

        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            discounted_reward = reward + (self.gamma * discounted_reward * (1 - is_terminal))
            rewards.insert(0, discounted_reward)

        rewards = torch.tensor(rewards).to(self.device).float().view(-1, 1)
        logprobs = torch.stack(self.memory.logprobs).to(self.device)
        values = torch.stack(self.memory.values)
        advantages = rewards.squeeze() - values.squeeze()
        # advantages_split=torch.split(advantages, tuple(lengths.flatten()))

        reinforce_loss = -logprobs.squeeze(dim=1) * advantages
        vf_loss = self.MSE_loss(values.squeeze(), rewards.squeeze())
        loss = reinforce_loss + vf_loss
        loss_per_step = torch.split(loss, tuple(lengths.flatten()))


        loss_per_episode = torch.stack([torch.sum(l) for l in loss_per_step])
        self.writer.add_scalar('reinforce_loss', reinforce_loss.mean(), self.writer_iteration + 1)
        self.writer.add_scalar('vf_loss', vf_loss.mean(), self.writer_iteration + 1)
        self.writer.add_scalar('loss', loss_per_episode.mean(), self.writer_iteration + 1)
        logging.info("values {}".format(values.mean()))
        # loss=torch.cat(losses)
        self.writer_iteration += 1

        # take gradient step
        self.optimizer.zero_grad()
        loss_per_episode.mean().backward()
        self.optimizer.step()
