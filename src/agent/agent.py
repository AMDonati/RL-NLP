import logging
import random

import torch
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_img = []
        self.states_text = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.states_img[:]
        del self.states_text[:]
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
        self.generated_text = []

    def get_top_k_words(self, state_text, state_img, top_k=10):
        """
        Truncate the action space with the top k words of a pretrained language model
        :param state: state
        :param top_k: number of words
        :return: top k words
        """
        if self.pretrained_lm is None:
            return None
        dist, value = self.pretrained_lm.act(state_text, state_img)
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

    def get_metrics(self, question):
        self.generated_text.append(question.view(-1)[1:].cpu().numpy())
        last_text = [item for sublist in self.generated_text[-min(10, len(self.generated_text)):] for item in sublist]
        diversity_metric = len(set(last_text)) / len(last_text)
        return diversity_metric

    def get_bleu_score(self, question):
        question_decoded = self.env.clevr_dataset.idx2word(question, stop_at_end=True)
        ref_questions = [q.split() for q in self.env.ref_questions_decoded]
        question_tokens = question_decoded.split()
        score = sentence_bleu(ref_questions, question_tokens)
        return score

    def test(self, log_interval=1, num_episodes=10):
        self.generated_text = []
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
                    if self.pretrained_lm != None:
                        top_k_indices = torch.gather(valid_actions, 1, top_k_indices)
                    top_words_decoded = self.env.clevr_dataset.idx2word(top_k_indices.cpu().numpy()[0])
                    weights_words = ["{}/{:.3f}".format(word, weight, number=3) for word, weight in
                                     zip(top_words_decoded.split(), top_k_weights[0].cpu().detach().numpy())]
                    top_words.append("next possible words for {} : {}".format(state_decoded, ", ".join(weights_words)))
                    if self.pretrained_lm is None:
                        target_word_log_prob = dist.log_prob(ref_question[t].to(self.device))
                    else:
                        if ref_question[t].to(self.device) in valid_actions:
                            target_word = list(valid_actions.view(-1).cpu().numpy()).index(ref_question[t])
                            target_word_log_prob = dist.log_prob(torch.tensor([target_word]).float().to(self.device))
                        else:
                            # case where the target word is not in the top words of the language model
                            target_word_log_prob = torch.tensor([-10]).float().to(self.device)
                    log_probs_ppl.append(target_word_log_prob)
                    idx_step += 1
                    state, (reward, _), done, _ = self.env.step(action)
                    ep_reward += reward
                    if done:
                        self.writer.add_scalar('test_TTR', self.get_metrics(state.text), i_episode + 1)
                        score = self.get_bleu_score(state.text)
                        #self.writer.add_scalar('test_BLEU', self.get_bleu_score(state.text), i_episode + 1)

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
                action, log_probs, value, _, _ = self.select_action(state=state,
                                                                    forced=forced,
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
                    self.writer.add_scalar('train_TTR', self.get_metrics(state.text), i_episode + 1)
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


