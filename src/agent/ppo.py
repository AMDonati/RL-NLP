import logging

import torch
import torch.nn as nn

from agent.reinforce import REINFORCE


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class PPO(REINFORCE):
    def __init__(self, policy, policy_old, env, gamma=1., lr=1e-2, pretrained_lm=None, update_timestep=50, K_epochs=10):
        REINFORCE.__init__(self, policy, env, gamma=gamma, lr=lr, pretrained_lm=pretrained_lm)
        self.policy_old = policy_old
        self.memory = Memory()
        self.update_timestep = update_timestep
        self.K_epoch = K_epochs
        self.MSE_loss = nn.MSELoss()
        self.eps_clip = 0.2
        self.K_epochs = K_epochs
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def select_action(self, state):
        # valid_actions = self.get_top_k_words(state, num_truncated)
        m, value = self.policy_old.act([state])
        action = m.sample()
        log_prob = m.log_prob(action).view(1)
        self.memory.states.append(state)
        self.memory.actions.append(action)
        self.memory.logprobs.append(log_prob)
        return action.item(), log_prob, value, None, m

    def evaluate(self, state, action):
        # valid_actions = self.get_top_k_words(state, num_truncated)
        m, value = self.policy.act(state)
        dist_entropy = m.entropy()

        # action = m.sample()
        actions=action.view(-1)
        log_prob = m.log_prob(actions)
        return log_prob, value, dist_entropy

    def update(self):

        # Monte Carlo estimate of state rewards:
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.memory.rewards), reversed(self.memory.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
        # rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-5)
        rewards = torch.tensor(rewards).to(self.device).float()

        # convert list to tensor
        # old_states = torch.stack(self.memory.states).to(self.device).detach()
        old_states = self.memory.states
        old_actions = torch.stack(self.memory.actions).to(self.device).detach()
        old_logprobs = torch.stack(self.memory.logprobs).to(self.device).detach()

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Evaluating old actions and values :
            logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

            # Finding the ratio (pi_theta / pi_theta__old):
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss:
            advantages = rewards - state_values.detach()
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            loss = -torch.min(surr1, surr2) + 0.5 * self.MSE_loss(state_values, rewards) - 0.01 * dist_entropy

            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()

        # Copy new weights into old policy:
        self.policy_old.load_state_dict(self.policy.state_dict())

    def learn(self, writer, output_path="lm", log_interval=10, num_episodes=100, pretrain=False,
              num_truncated=10):

        running_reward = 0
        timestep = 0
        for i_episode in range(num_episodes):
            state, ep_reward = self.env.reset(), 0
            for t in range(0, self.env.max_len + 1):
                action, log_probs, value, _, _ = self.select_action(state)
                state, (reward, _), done, _ = self.env.step(action)
                # Saving reward and is_terminal:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)

                timestep+=1

                # update if its time
                if timestep % self.update_timestep == 0:
                    self.update()
                    self.memory.clear_memory()
                    timestep = 0


                ep_reward += reward
                if done:
                    break
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            if i_episode % log_interval == 0:
                logging.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
                writer.add_text('episode_questions', '  \n'.join(self.env.ref_questions_decoded))
                writer.add_scalar('train_running_return', running_reward, i_episode + 1)
