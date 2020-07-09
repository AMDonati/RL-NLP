import logging

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from agent.agent import Agent


class PPO(Agent):
    def __init__(self, policy, env, writer, gamma=1., eps_clip=0.2, pretrained_lm=None, lm_sl=True,
                 pretrained_policy=None,
                 update_every=100, K_epochs=10, entropy_coeff=0.01, pretrain=False, word_emb_size=8, hidden_size=24,
                 kernel_size=1,
                 stride=2, num_filters=3, num_truncated=10, truncate_mode="masked"):
        Agent.__init__(self, policy, env, writer, gamma=gamma, pretrained_lm=pretrained_lm, lm_sl=lm_sl,
                       pretrained_policy=pretrained_policy, pretrain=pretrain, update_every=update_every,
                       word_emb_size=word_emb_size, hidden_size=hidden_size, kernel_size=kernel_size, stride=stride,
                       num_filters=num_filters, num_truncated=num_truncated, truncate_mode=truncate_mode)
        self.policy_old = policy(env.clevr_dataset.len_vocab, word_emb_size, hidden_size, kernel_size=kernel_size,
                                 stride=stride, num_filters=num_filters, rl=True, truncate_mode=truncate_mode)
        self.policy_old.load_state_dict(self.policy.state_dict())
        self.policy_old.to(self.device)
        self.K_epochs = K_epochs
        self.MSE_loss = nn.MSELoss(reduction="none")
        self.eps_clip = eps_clip
        self.entropy_coeff = entropy_coeff
        self.update_mode = "episode"
        self.writer_iteration = 0

    def select_action(self, state, num_truncated=10, forced=None):
        valid_actions, actions_probs = self.get_top_k_words(state.text, num_truncated, state.img)
        policy_dist, policy_dist_truncated, value = self.policy_old(state.text, state.img, valid_actions)
        action = policy_dist_truncated.sample() if forced is None else forced
        if policy_dist_truncated.probs.size() != policy_dist.probs.size():
            action = torch.gather(valid_actions, 1, action.view(1, 1))
        log_prob = policy_dist.log_prob(action.to(self.device)).view(-1)
        self.memory.actions.append(action)
        self.memory.states_img.append(state.img[0])
        self.memory.states_text.append(state.text[0])
        self.memory.logprobs.append(log_prob)
        return action.cpu().numpy(), log_prob, value, (valid_actions, actions_probs), policy_dist

    def evaluate(self, state_text, state_img, action, num_truncated=10):
        valid_actions, actions_probs = self.get_top_k_words(state_text, num_truncated)
        policy_dist, policy_dist_truncated, value = self.policy(state_text, state_img, valid_actions)
        dist_entropy = policy_dist.entropy()
        log_prob = policy_dist.log_prob(action.view(-1))

        return log_prob, value, dist_entropy

    def update(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        rewards = torch.tensor(rewards).to(self.device).float()

        old_states_text = pad_sequence(self.memory.states_text, batch_first=True, padding_value=0).to(self.device)
        old_states_img = torch.stack(self.memory.states_img)
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.evaluate(old_states_text, old_states_img, old_actions,
                                                                 self.num_truncated)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach().view(-1))

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach().squeeze() if not self.pretrain else 1
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            surr = -torch.min(surr1, surr2)
            # entropy_loss = self.entropy_coeff * torch.tensor(entropy_coeffs) * dist_entropy
            entropy_loss = self.entropy_coeff * dist_entropy

            vf_loss = 0.5 * self.MSE_loss(state_values.squeeze(), rewards) if not self.pretrain else torch.tensor(
                [0]).float().to(
                self.device)
            loss = surr + vf_loss - entropy_loss
            logging.info(
                "loss {} entropy {} surr {} mse {} ".format(loss.mean(), dist_entropy.mean(),
                                                            surr.mean(),
                                                            vf_loss.mean()))

            self.writer.add_scalar('loss', loss.mean(), self.writer_iteration + 1)
            self.writer.add_scalar('entropy', dist_entropy.mean(), self.writer_iteration + 1)
            self.writer.add_scalar('vf_loss', vf_loss.mean(), self.writer_iteration + 1)
            self.writer.add_scalar('surrogate', surr.mean(), self.writer_iteration + 1)
            self.writer.add_scalar('ratios', ratios.mean(), self.writer_iteration + 1)

            self.writer_iteration += 1
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())
