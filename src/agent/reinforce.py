import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from RL_toolbox.RL_functions import compute_grad_norm
from agent.agent import Agent


class REINFORCE(Agent):
    def __init__(self, policy, env, test_envs, pretrained_lm, writer, out_path, gamma=1., lr=1e-2, grad_clip=None,
                 pretrain=False, update_every=50, num_truncated=10, p_th=None, truncate_mode="top_k", log_interval=10,
                 eval_no_trunc=0, alpha_logits=0., alpha_decay_rate=0., epsilon_truncated=0., train_seed=0,
                 epsilon_truncated_rate=1.,
                 is_loss_correction=1, train_metrics=[], test_metrics=[], top_p=1.):
        Agent.__init__(self, policy=policy, env=env, writer=writer, out_path=out_path, gamma=gamma, lr=lr,
                       grad_clip=grad_clip,
                       pretrained_lm=pretrained_lm,
                       pretrain=pretrain, update_every=update_every,
                       num_truncated=num_truncated,
                       p_th=p_th,
                       truncate_mode=truncate_mode,
                       log_interval=log_interval, test_envs=test_envs, eval_no_trunc=eval_no_trunc,
                       alpha_logits=alpha_logits, alpha_decay_rate=alpha_decay_rate,
                       epsilon_truncated=epsilon_truncated,
                       train_seed=train_seed, epsilon_truncated_rate=epsilon_truncated_rate,
                       is_loss_correction=is_loss_correction, train_metrics=train_metrics, test_metrics=test_metrics,
                       top_p=top_p)
        self.MSE_loss = nn.MSELoss(reduction="none")
        self.grad_clip = grad_clip
        self.update_mode = "episode"
        self.writer_iteration = 0

    def evaluate(self, state_text, state_img, states_answer, action):
        policy_dist, policy_dist_truncated, value = self.policy(state_text, state_img, states_answer,
                                                                valid_actions=None)
        dist_entropy = policy_dist.entropy()
        log_prob = policy_dist.log_prob(action.view(-1))
        return log_prob, value, dist_entropy

    def split_memory_per_episode(self, element):
        is_terminals = np.array(self.memory.is_terminals)
        lengths = np.argwhere(is_terminals == True) + 1
        lengths[1:] = lengths[1:] - lengths[:-1]
        episode_tensors = torch.split(element, tuple(lengths.flatten()))
        return episode_tensors

    def compute_returns(self):
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        return rewards

    def update(self):
        returns = self.compute_returns()
        returns = torch.tensor(returns).to(self.device).float()

        logprobs = torch.stack(self.memory.logprobs).to(self.device)
        logprobs_truncated = torch.stack(self.memory.logprobs_truncated).to(self.device)
        values = torch.stack(self.memory.values).to(self.device)

        advantages = returns - values.detach().squeeze()
        rl_loss_per_timestep = -logprobs.view(-1) * advantages
        rl_loss_per_timestep = self.split_memory_per_episode(rl_loss_per_timestep)
        is_ratios = torch.exp(logprobs.detach() - logprobs_truncated.detach())
        is_ratios = self.split_memory_per_episode(is_ratios)
        if self.is_loss_correction:
            rl_loss_per_episode = torch.stack(
                [torch.sum(l) * torch.prod(p) for l, p in zip(rl_loss_per_timestep, is_ratios)])
        else:
            rl_loss_per_episode = torch.stack(
                [torch.sum(l) for l in rl_loss_per_timestep])
        reinforce_loss = rl_loss_per_episode.mean()

        vf_loss = 0.5 * self.MSE_loss(values.view(-1), returns).mean()

        loss = reinforce_loss + vf_loss  # TODO: add an entropy term here as well.
        # take gradient step
        self.optimizer.zero_grad()
        loss.backward()
        # clip grad norm:
        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(parameters=self.policy.parameters(), max_norm=self.grad_clip)
        self.optimizer.step()
        # compute grad norm:
        grad_norm = compute_grad_norm(self.policy)
        self.writer.add_scalar('grad_norm', grad_norm, self.writer_iteration + 1)
        self.writer_iteration += 1

        # compute new log_probs for comparison with old ones:
        states_text = pad_sequence(self.memory.states_text, batch_first=True, padding_value=0).to(self.device)
        policy_dist, policy_dist_truncated, value = self.policy(states_text, torch.stack(self.memory.states_img),
                                                                torch.stack(self.memory.states_answer))
        new_probs = torch.gather(policy_dist.probs, 1, torch.stack(self.memory.actions))
        ratios = torch.exp(torch.log(new_probs) - logprobs).detach()
        self.writer.add_scalar('ratios', ratios.mean(), self.writer_iteration + 1)
        self.writer.add_scalar('rl_loss', reinforce_loss, self.writer_iteration + 1)

        return loss.mean()
