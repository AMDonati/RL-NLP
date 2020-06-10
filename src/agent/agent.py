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
        self.values = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]


class Agent:
    def __init__(self, policy, env, writer, gamma=1., lr=1e-2, pretrained_lm=None, pretrain=False,
                 update_every=50, word_emb_size=8, hidden_size=24, kernel_size=1, stride=2, num_filters=3,
                 num_truncated=10):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy(env.clevr_dataset.len_vocab, word_emb_size, hidden_size, kernel_size=kernel_size,
                             stride=stride, num_filters=num_filters)
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.pretrained_lm = pretrained_lm
        self.env = env
        self.pretrain = pretrain
        self.update_every = update_every
        self.memory = Memory()
        self.num_truncated = num_truncated
        self.writer = writer

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
        return top_k_indices

    def select_action(self, state, forced=None, num_truncated=10):
        pass

    def finish_episode(self):
        pass

    # def learn(self, env, writer, output_path="lm", log_interval=10, num_episodes=100, pretrain=False,
    #          num_truncated=10):
    #    pass

    def save(self, out_file):
        torch.save(self.policy, out_file)

    def test(self, log_interval=1, num_episodes=10):
        # trained_model.load_state_dict(torch.load(saved_path))
        log_probs_ppl = []
        self.policy.eval()
        running_reward, idx_step = 0, 0
        for i_episode in range(num_episodes):
            top_words = []
            for ref_question in self.env.ref_questions:
                state, ep_reward = self.env.reset(), 0
                for t in range(0, self.env.max_len):
                    action, log_probs, value, valid_actions, dist = self.select_action(state=state,
                                                                                       num_truncated=self.num_truncated)
                    state_decoded = self.env.clevr_dataset.idx2word(state.text.numpy()[0])
                    top_k_weights, top_k_indices = torch.topk(dist.probs, self.num_truncated, sorted=True)
                    top_words_decoded = self.env.clevr_dataset.idx2word(top_k_indices.cpu().numpy()[0])
                    # top = " ".join(
                    #    ["{}/{}".format(token, weight) for token, weight in zip(top_words_decoded.split(), top_k_weights.numpy())])
                    weights_words = ["{}/{:.3f}".format(word, weight, number=3) for word, weight in
                                     zip(top_words_decoded.split(), top_k_weights[0].detach().numpy())]
                    top_words.append("next possible words for {} : {}".format(state_decoded, ", ".join(weights_words)))
                    if self.pretrained_lm is None:
                        target_word_log_prob = dist.log_prob(ref_question[t].to(self.device))
                    else:
                        valid_actions = self.get_top_k_words([state], self.num_truncated)
                        if ref_question[t] in valid_actions:
                            target_word = list(valid_actions.view(-1).numpy()).index(ref_question[t])
                            target_word_log_prob = dist.log_prob(torch.tensor([target_word]).float().to(self.device))
                        else:
                            target_word_log_prob = torch.tensor([-10]).float().to(self.device)
                    log_probs_ppl.append(target_word_log_prob)
                    idx_step += 1
                    state, (reward, _), done, _ = self.env.step(action)
                    ep_reward += reward
                    if done:
                        break
            ppl = torch.exp(-torch.stack(log_probs_ppl).sum() / idx_step)
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            if i_episode % log_interval == 0:
                logging.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
                self.writer.add_text('episode_questions', '  \n'.join(self.env.ref_questions_decoded))
                self.writer.add_scalar('test_running_return', running_reward, i_episode + 1)
                self.writer.add_scalar('ppl', ppl, i_episode + 1)
                self.writer.add_text('language_model', '  \n'.join(top_words))

    def learn(self, log_interval=10, num_episodes=100):

        running_reward = 0
        timestep = 0
        for i_episode in range(num_episodes):
            state, ep_reward = self.env.reset(), 0
            ref_question = random.choice(self.env.ref_questions)
            for t in range(0, self.env.max_len):
                forced = ref_question[t] if self.pretrain else None
                action, log_probs, value, _, _ = self.select_action(state=state, forced=forced,
                                                                    num_truncated=self.num_truncated)
                state, (reward, _), done, _ = self.env.step(action)
                # Saving reward and is_terminal:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)

                timestep += 1

                # update if its time
                if self.update_mode == "step" and timestep % self.update_every == 0:
                    self.update()
                    self.memory.clear_memory()
                    timestep = 0

                ep_reward += reward
                if done:
                    if self.update_mode == "episode" and i_episode % self.update_every == 0:
                        self.update()
                        self.memory.clear_memory()
                    break
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            if i_episode % log_interval == 0:
                logging.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
                self.writer.add_text('episode_questions', '  \n'.join(self.env.ref_questions_decoded))
                self.writer.add_scalar('train_running_return', running_reward, i_episode + 1)
