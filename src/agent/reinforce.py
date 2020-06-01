import logging
import os
import random

import pandas as pd
import torch
import torch.nn as nn

import agent
from agent.agent import Agent


class REINFORCE(Agent):
    def __init__(self, model, env, gamma=1., lr=1e-2, pretrained_lm=None):
        Agent.__init__(self, model, env, gamma=gamma, lr=lr, pretrained_lm=pretrained_lm)

    def get_top_k_words(self, state, top_k=10):
        """
        Truncate the action space with the top k words of a pretrained language model
        :param state: state
        :param top_k: number of words
        :return: top k words
        """
        if self.pretrained_lm is None:
            return None
        dist, value = self.pretrained_lm(state.text, state.img, None)
        probs = dist.probs
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
        valid_actions = {i: token for i, token in enumerate(top_k_indices.numpy()[0])}
        return valid_actions

    def select_action(self, state, forced=None, num_truncated=10):
        valid_actions = self.get_top_k_words(state, num_truncated)
        m, value = self.model(state.text, state.img, valid_actions)
        action = m.sample() if forced is None else forced
        log_prob = m.log_prob(action).view(1)
        if isinstance(valid_actions, dict):
            action = torch.tensor(valid_actions[action.item()]).view(1)
        return action.item(), log_prob, value, valid_actions, m

    def finish_episode(self):
        R = 0
        policy_loss = []
        returns = []
        mse = nn.MSELoss()
        for r in self.model.rewards[::-1]:
            R = r + self.gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns).float()
        for log_prob, R, value in zip(self.model.saved_log_probs, returns, self.model.values):
            policy_loss.append(-log_prob * (R - value))
            ms = mse(value, R)
            # ms = mse(value, torch.tensor(self.max_reward).float())
            policy_loss.append(ms.view(1))
        self.optimizer.zero_grad()
        policy_loss = torch.cat(policy_loss).sum()
        policy_loss.backward()
        self.optimizer.step()
        del self.model.rewards[:]
        del self.model.saved_log_probs[:]
        del self.model.values[:]

    def learn(self, env, writer, output_path="lm", log_interval=10, num_episodes=100, pretrain=False,
              num_truncated=10):
        running_reward = 0
        for i_episode in range(num_episodes):
            state, ep_reward = env.reset(), 0
            ref_question = env.ref_questions[random.randint(0, len(env.ref_questions) - 1)]
            for t in range(0, env.max_len + 1):
                forced = ref_question[t] if pretrain else None
                action, log_probs, value, _, _ = self.select_action(state, forced=forced, num_truncated=num_truncated)
                state, (reward, _), done, _ = self.env.step(action)
                if pretrain:
                    value = torch.tensor([-1.])
                self.model.rewards.append(reward)
                self.model.values.append(value)
                self.model.saved_log_probs.append(log_probs)
                ep_reward += reward
                if done:
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            self.finish_episode()
            if i_episode % log_interval == 0:
                logging.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
                writer.add_text('episode_questions', '  \n'.join(env.ref_questions_decoded))
                writer.add_scalar('train_running_return', running_reward, i_episode + 1)

            #df = pd.DataFrame(self.model.last_policy[-env.max_len:])
            # diff_df = df.diff(periods=5)
            #diff_df = (df.iloc[-1] - df.iloc[0]).abs()
            #top_words = diff_df.nlargest(4)
            #logging.info("top words changed in the policy : {}".format(env.clevr_dataset.idx2word(top_words.index)))

        out_file = os.path.join(output_path, 'model.pth')
        # with open(out_file, 'wb') as f:
        #    torch.save(agent.model, f)
        torch.save(self.model.state_dict(), out_file)
        return agent, out_file

