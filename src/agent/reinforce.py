import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from RL_toolbox.RL_functions import compute_grad_norm
from agent.agent import Agent


class REINFORCE(Agent):
    def __init__(self, policy, env, test_envs, pretrained_lm, writer, out_path, gamma=1., lr=1e-2, grad_clip=None,
                 pretrain=False, update_every=50, num_truncated=10, p_th=None, truncate_mode="top_k", log_interval=10,
                 eval_no_trunc=0, alpha_logits=0., alpha_decay_rate=0., epsilon_truncated=0.):
        Agent.__init__(self, policy=policy, env=env, writer=writer, out_path=out_path, gamma=gamma, lr=lr,
                       grad_clip=grad_clip,
                       pretrained_lm=pretrained_lm,
                       pretrain=pretrain, update_every=update_every,
                       num_truncated=num_truncated,
                       p_th=p_th,
                       truncate_mode=truncate_mode,
                       log_interval=log_interval, test_envs=test_envs, eval_no_trunc=eval_no_trunc,
                       alpha_logits=alpha_logits, alpha_decay_rate=alpha_decay_rate,
                       epsilon_truncated=epsilon_truncated)
        self.MSE_loss = nn.MSELoss(reduction="none")
        self.grad_clip = grad_clip
        self.update_mode = "episode"
        self.writer_iteration = 0

    def evaluate(self, state_text, state_img, action):
        policy_dist, policy_dist_truncated, value = self.policy(state_text, state_img, valid_actions=None)
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

        logprobs = torch.stack(self.memory.logprobs).to(self.device)
        values = torch.stack(self.memory.values).to(self.device)

        advantages = rewards - values.detach().squeeze() if not self.pretrain else 1
        reinforce_loss = -logprobs.view(-1) * advantages
        vf_loss = 0.5 * self.MSE_loss(values.view(-1), rewards) if not self.pretrain else torch.tensor(
            [0]).float().to(
            self.device)
        loss = reinforce_loss + vf_loss
        loss = loss.sum() / self.update_every
        # loss = loss.mean()
        # take gradient step
        self.optimizer.zero_grad()
        # loss.sum().backward()
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
        policy_dist, policy_dist_truncated, value = self.policy(states_text, torch.stack(self.memory.states_img))
        new_probs = torch.gather(policy_dist.probs, 1, torch.stack(self.memory.actions))
        ratios = torch.exp(torch.log(new_probs) - logprobs)
        self.writer.add_scalar('ratios', ratios.mean(), self.writer_iteration + 1)

        return loss.mean()
