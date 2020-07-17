import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from agent.agent import Agent


class REINFORCE(Agent):
    def __init__(self, policy, env, test_envs, writer, out_path, gamma=1., lr=1e-2, grad_clip=None, pretrained_lm=None,
                 lm_sl=True,
                 pretrain=False, update_every=50, num_truncated=10, log_interval=10):
        Agent.__init__(self, policy, env, writer, out_path, gamma=gamma, lr=lr, grad_clip=grad_clip,
                       pretrained_lm=pretrained_lm,
                       lm_sl=lm_sl,
                       pretrain=pretrain, update_every=update_every,
                       num_truncated=num_truncated,
                       log_interval=log_interval, test_envs=test_envs)
        self.MSE_loss = nn.MSELoss(reduction="none")
        self.grad_clip = grad_clip
        self.update_mode = "episode"
        self.writer_iteration = 0

    def select_action(self, state, num_truncated=10, forced=None):
        valid_actions, actions_probs = self.get_top_k_words(state.text, num_truncated, state.img)
        policy_dist, policy_dist_truncated, value = self.policy(state.text, state.img, valid_actions)
        action = policy_dist.sample() if forced is None else forced
        if policy_dist_truncated.probs.size() != policy_dist.probs.size():
            action = torch.gather(valid_actions, 1, action.view(1, 1))
        log_prob = policy_dist.log_prob(action.to(self.device)).view(-1)
        return action, log_prob, value, (valid_actions, actions_probs), policy_dist

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
        # rewards=torch.tensor(self.memory.rewards).to(self.device).float()

        old_states_text = pad_sequence(self.memory.states_text, batch_first=True, padding_value=0).to(self.device)
        old_states_img = torch.stack(self.memory.states_img)
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device)
        old_values = torch.stack(self.memory.values).to(self.device)

        # Evaluating old actions and values :
        #logprobs, state_values, dist_entropy = self.evaluate(old_states_text, old_states_img, old_actions,
        #                                                     self.num_truncated)

        # Finding Surrogate Loss:
        advantages = rewards - old_values.detach().squeeze() if not self.pretrain else 1
        reinforce_loss = -old_logprobs.view(-1)*advantages
        vf_loss = 0.5 * self.MSE_loss(old_values.view(-1), rewards) if not self.pretrain else torch.tensor(
            [0]).float().to(
            self.device)
        loss = reinforce_loss + vf_loss
        # logging.info(
        #     "loss {} entropy {} surr {} mse {} ".format(loss.mean(), dist_entropy.mean(),
        #                                                 surr.mean(),
        #                                                 vf_loss.mean()))

        # take gradient step
        self.optimizer.zero_grad()
        loss.sum().backward()
        # clip grad norm:
        self.optimizer.step()
        # compute grad norm:
        self.writer_iteration += 1

        # Copy new weights into old policy:

        return loss.mean()


