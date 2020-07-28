import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from agent.agent import Agent
from RL_toolbox.RL_functions import compute_grad_norm


class REINFORCE(Agent):
    def __init__(self, policy, env, test_envs, writer, out_path, gamma=1., lr=1e-2, eps=1e-08, grad_clip=None, pretrained_lm=None,
                 lm_sl=True,
                 pretrain=False, update_every=50, num_truncated=10, truncate_mode="masked",log_interval=10):
        Agent.__init__(self, policy, env, writer, out_path, gamma=gamma, lr=lr, eps=eps, grad_clip=grad_clip,
                       pretrained_lm=pretrained_lm,
                       lm_sl=lm_sl,
                       pretrain=pretrain, update_every=update_every,
                       num_truncated=num_truncated,
                       truncate_mode=truncate_mode,
                       log_interval=log_interval, test_envs=test_envs)
        self.MSE_loss = nn.MSELoss(reduction="none")
        self.grad_clip = grad_clip
        self.update_mode = "episode"
        self.writer_iteration = 0

    def select_action(self, state, num_truncated=10, forced=None):
        valid_actions, actions_probs = self.get_top_k_words(state.text, num_truncated, state.img)
        policy_dist, policy_dist_truncated, value = self.policy(state.text, state.img, valid_actions)
        action = policy_dist_truncated.sample() if forced is None else forced
        if policy_dist_truncated.probs.size() != policy_dist.probs.size():
            action = torch.gather(valid_actions, 1, action.view(1, 1))
        log_prob = policy_dist.log_prob(action.to(self.device)).view(-1)
        log_prob_truncated = policy_dist_truncated.log_prob(action.to(self.device)).view(-1)
        return action, log_prob, value, (valid_actions, actions_probs, log_prob_truncated), policy_dist

    def evaluate(self, state_text, state_img, action, num_truncated=10):
        #valid_actions, actions_probs = self.get_top_k_words(state_text, num_truncated)
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
        reinforce_loss = -logprobs.view(-1)*advantages
        vf_loss = 0.5 * self.MSE_loss(values.view(-1), rewards) if not self.pretrain else torch.tensor(
            [0]).float().to(
            self.device)
        loss = reinforce_loss + vf_loss
        loss = loss.sum() / self.update_every
        #loss = loss.mean()
        # take gradient step
        self.optimizer.zero_grad()
        #loss.sum().backward()
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


