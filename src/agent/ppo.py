import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence

from RL_toolbox.RL_functions import compute_grad_norm
from agent.agent import Agent


class PPO(Agent):
    def __init__(self, policy, env, test_envs, pretrained_lm, writer, out_path, gamma=1., lr=1e-2, eps_clip=0.2,
                 grad_clip=None,
                 truncate_mode="top_k",
                 update_every=100, num_truncated=10,
                 p_th=None,
                 K_epochs=10, entropy_coeff=0.01, pretrain=False,
                 log_interval=1,
                 eval_no_trunc=0,
                 alpha_logits=0.,
                 alpha_decay_rate=0.,
                 epsilon_truncated=0.,
                 train_seed=0,
                 epsilon_truncated_rate=1.,
                 is_loss_correction=1, train_metrics=[], test_metrics=[]):
        Agent.__init__(self, policy=policy, env=env, writer=writer, pretrained_lm=pretrained_lm, out_path=out_path,
                       gamma=gamma, lr=lr,
                       grad_clip=grad_clip,
                       pretrain=pretrain, update_every=update_every,
                       num_truncated=num_truncated,
                       p_th=p_th,
                       truncate_mode=truncate_mode,
                       log_interval=log_interval, test_envs=test_envs,
                       eval_no_trunc=eval_no_trunc,
                       alpha_logits=alpha_logits, alpha_decay_rate=alpha_decay_rate,
                       epsilon_truncated=epsilon_truncated,
                       train_seed=train_seed,
                       epsilon_truncated_rate=epsilon_truncated_rate,
                       is_loss_correction=is_loss_correction, train_metrics=train_metrics, test_metrics=test_metrics)
        self.policy_old = policy
        self.policy_old.to(self.device)
        self.K_epochs = K_epochs
        self.MSE_loss = nn.MSELoss(reduction="none")
        self.eps_clip = eps_clip
        self.grad_clip = grad_clip
        self.entropy_coeff = entropy_coeff
        self.update_mode = "episode"
        self.writer_iteration = 0

    def evaluate(self, state_text, state_img, states_answer, action):
        policy_dist, _, value = self.policy(state_text, state_img, states_answer, valid_actions=None)
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
        old_states_answer = torch.stack(self.memory.states_answer)
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()
        old_logprobs_truncated = torch.stack(self.memory.logprobs_truncated).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values:
            logprobs, state_values, dist_entropy = self.evaluate(old_states_text, old_states_img, old_states_answer,
                                                                 old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach().view(-1))

            # adding the is_ratio:
            if self.is_loss_correction and self.truncate_mode is not None:
                sampling_term = old_logprobs_truncated * (
                        1 - self.epsilon_truncated) + self.epsilon_truncated * old_logprobs
                # computing the Importance Sampling ratio (pi_theta_old / rho_theta_old)
                is_ratios = torch.exp(old_logprobs - sampling_term.to(self.device)).view(-1)
                ratios = ratios * is_ratios

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach().squeeze() if not self.pretrain else 1
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            surr = -torch.min(surr1, surr2)
            entropy_loss = self.entropy_coeff * dist_entropy

            vf_loss = 0.5 * self.MSE_loss(state_values.squeeze(), rewards)
            loss = surr + vf_loss - entropy_loss

            self.writer.add_scalar('loss', loss.mean(), self.writer_iteration + 1)
            self.writer.add_scalar('loss_vf', vf_loss.mean(), self.writer_iteration + 1)
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

    def get_policy_distributions(self, state, valid_actions, logits_lm=None, alpha=0., baseline=False):
        policy = self.start_policy if baseline else self.policy_old
        policy_dist, policy_dist_truncated, value = policy(state.text, state.img, state.answer,
                                                           valid_actions=valid_actions,
                                                           logits_lm=logits_lm, alpha=alpha)
        return policy_dist, policy_dist_truncated, value
