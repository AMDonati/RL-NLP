import logging
import random

import torch
import torch.optim as optim


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


class Agent:
    def __init__(self, model, env, gamma=1., lr=1e-2, pretrained_lm=None):
        self.policy = model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.pretrained_lm = pretrained_lm
        self.env = env

    def get_top_k_words(self, state, top_k=10):
        """
        Truncate the action space with the top k words of a pretrained language model
        :param state: state
        :param top_k: number of words
        :return: top k words
        """
        if self.pretrained_lm is None:
            return None
        dist, value = self.pretrained_lm.act(state)
        probs = dist.probs
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
        #valid_actions = {i: token for i, token in enumerate(top_k_indices.numpy())}
        return top_k_indices

    def select_action(self, state, forced=None, num_truncated=10):
        pass

    def finish_episode(self):
        pass

    #def learn(self, env, writer, output_path="lm", log_interval=10, num_episodes=100, pretrain=False,
    #          num_truncated=10):
    #    pass

    def save(self, out_file):
        torch.save(self.policy, out_file)

    def learn(self, writer, output_path="lm", log_interval=10, num_episodes=100, pretrain=False,
              num_truncated=10):

        running_reward = 0
        timestep = 0
        for i_episode in range(num_episodes):
            state, ep_reward = self.env.reset(), 0
            ref_question = random.choice(self.env.ref_questions)
            for t in range(0, self.env.max_len + 1):
                forced = ref_question[t] if self.pretrain else None
                action, log_probs, value, _, _ = self.select_action(state=state, forced=forced)
                state, (reward, _), done, _ = self.env.step(action)
                # Saving reward and is_terminal:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)

                timestep += 1

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

    def test(self, writer,log_interval=1, num_episodes=10):
        #trained_model.load_state_dict(torch.load(saved_path))
        self.policy.eval()

        running_reward = 0
        for i_episode in range(num_episodes):
            state, ep_reward = self.env.reset(), 0
            top_words = []
            for t in range(0, self.env.max_len + 1):
                action, log_probs, value, valid_actions, dist = self.select_action(state)
                state_decoded = self.env.clevr_dataset.idx2word(state.text.numpy()[0])
                top_k_weights, top_k_indices = torch.topk(dist.probs, 10, sorted=True)
                top_words_decoded = self.env.clevr_dataset.idx2word(top_k_indices.numpy()[0])
                # top = " ".join(
                #    ["{}/{}".format(token, weight) for token, weight in zip(top_words_decoded.split(), top_k_weights.numpy())])
                top_words.append("next 10 possible words for {} : {}".format(state_decoded, top_words_decoded))
                state, (reward, _), done, _ = self.env.step(action)
                ep_reward += reward
                if done:
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            if i_episode % log_interval == 0:
                logging.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
                writer.add_text('episode_questions', '  \n'.join(self.env.ref_questions_decoded))
                writer.add_scalar('test_running_return', running_reward, i_episode + 1)
                writer.add_text('language_model', '  \n'.join(top_words))


    def learn(self, writer, output_path="lm", log_interval=10, num_episodes=100, pretrain=False,
              num_truncated=10):

        running_reward = 0
        timestep = 0
        for i_episode in range(num_episodes):
            state, ep_reward = self.env.reset(), 0
            ref_question = random.choice(self.env.ref_questions)
            for t in range(0, self.env.max_len + 1):
                forced = ref_question[t] if self.pretrain else None
                action, log_probs, value, _, _ = self.select_action(state=state, forced=forced)
                state, (reward, _), done, _ = self.env.step(action)
                # Saving reward and is_terminal:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)

                timestep += 1

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
