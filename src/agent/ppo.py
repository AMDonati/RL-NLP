import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from RL_toolbox.RL_functions import compute_grad_norm
from agent.agent import Agent


class PPO(Agent):
    def __init__(self, policy, env, test_envs, pretrained_lm, writer, out_path, gamma=1., lr=1e-2, eps_clip=0.2, grad_clip=None,
                 truncate_mode="top_k",
                 update_every=100, num_truncated=10,
                 p_th=None,
                 K_epochs=10, entropy_coeff=0.01, pretrain=False,
                 log_interval=1,
                 eval_no_trunc=0,
                 lm_bonus=0):
        Agent.__init__(self, policy=policy, env=env, writer=writer, pretrained_lm=pretrained_lm, out_path=out_path, gamma=gamma, lr=lr,
                       grad_clip=grad_clip,
                       pretrain=pretrain, update_every=update_every,
                       num_truncated=num_truncated,
                       p_th=p_th,
                       truncate_mode=truncate_mode,
                       log_interval=log_interval, test_envs=test_envs,
                       eval_no_trunc=eval_no_trunc,
                       lm_bonus=lm_bonus)
        self.policy_old = policy
        self.policy_old.to(self.device)
        self.K_epochs = K_epochs
        self.MSE_loss = nn.MSELoss(reduction="none")
        self.eps_clip = eps_clip
        self.grad_clip = grad_clip
        self.entropy_coeff = entropy_coeff
        self.update_mode = "episode"
        self.writer_iteration = 0

    def evaluate(self, state_text, state_img, action, num_truncated=10):
        #valid_actions, actions_probs = self.get_top_k_words(state_text, num_truncated) #not needed and actually has a bug for batch of valid actions in LSTM models.
        policy_dist, _, value = self.policy(state_text, state_img, valid_actions=None)
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
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
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
            # logging.info(
            #     "loss {} entropy {} surr {} mse {} ".format(loss.mean(), dist_entropy.mean(),
            #                                                 surr.mean(),
            #                                                 vf_loss.mean()))

            self.writer.add_scalar('loss', loss.mean(), self.writer_iteration + 1)
            #self.writer.add_scalar('entropy', dist_entropy.mean(), self.writer_iteration + 1)
            self.writer.add_scalar('loss_vf', vf_loss.mean(), self.writer_iteration + 1)
            #self.writer.add_scalar('surrogate', surr.mean(), self.writer_iteration + 1)
            self.writer.add_scalar('ratios', ratios.mean(), self.writer_iteration + 1)

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            # clip grad norm:
            if self.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(parameters=self.policy.parameters(), max_norm=self.grad_clip)
            self.optimizer.step()
            # compute grad norm:
            grad_norm = compute_grad_norm(self.policy)
            self.writer.add_scalar('grad_norm', grad_norm, self.writer_iteration + 1)
            self.writer_iteration += 1

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

        return loss.mean()
