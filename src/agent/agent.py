import logging
import random

import torch
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu

from eval.metric import RewardMetric, DialogMetric, VAMetric, LMVAMetric


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
    def __init__(self, policy, env, writer, gamma=1., lr=1e-2, grad_clip=None, pretrained_lm=None, lm_sl=True, pretrained_policy=None,
                 pretrain=False, update_every=50, word_emb_size=8, hidden_size=24, kernel_size=1, stride=2,
                 num_filters=3, num_truncated=10):

        # torch.autograd.set_detect_anomaly(True)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy(env.clevr_dataset.len_vocab, word_emb_size, hidden_size, kernel_size=kernel_size,
                             stride=stride, num_filters=num_filters, rl=True)
        if pretrained_policy is not None:
            self.policy.load_state_dict(torch.load(pretrained_policy, map_location=self.device), strict=False)

        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr) #TODO: learning rate plays as well.
        self.gr
        self.gamma = gamma
        self.pretrained_lm = pretrained_lm
        self.lm_sl = lm_sl
        self.env = env
        self.pretrain = pretrain
        self.update_every = update_every
        self.memory = Memory()
        self.num_truncated = num_truncated
        self.writer = writer
        self.generated_text = []
        # self.metrics = [PPLMetric(self), RewardMetric(self), LMMetric(self), DialogMetric(self)]
        # self.test_metrics = [RewardMetric(self, train_test="test"), LMMetric(self, train_test="test"), DialogMetric(self, train_test="test")]
        self.test_metrics = [RewardMetric(self, train_test="test"),
                             DialogMetric(self, train_test="test")]
        self.train_metrics = [DialogMetric(self, train_test="train"), VAMetric(self, train_test="train"),
                              LMVAMetric(self, "train"), VAMetric(self, "train")]

    def get_top_k_words(self, state_text, top_k=10):
        """
        Truncate the action space with the top k words of a pretrained language model
        :param state: state
        :param top_k: number of words
        :return: top k words
        """
        if self.lm_sl:
            seq_len = state_text.size(1)
            if self.pretrained_lm is None:
                return None, None
            log_probas, _ = self.pretrained_lm(state_text)
            log_probas = log_probas.view(len(state_text), seq_len, -1)
            log_probas = log_probas[:, -1, :]
            top_k_weights, top_k_indices = torch.topk(log_probas, top_k, sorted=True)
        else:
            dist, value = self.pretrained_lm(state_text)
            probs = dist.probs
            top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

        return top_k_indices, top_k_weights

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
        question_decoded = self.env.clevr_dataset.idx2word(question.squeeze().cpu().numpy().tolist(), ignored=["<SOS>"],
                                                           stop_at_end=True)
        ref_questions = [q.split() for q in self.env.ref_questions_decoded]
        question_tokens = question_decoded.split()
        score = sentence_bleu(ref_questions, question_tokens)
        return score

    def test(self, log_interval=1, num_episodes=10):
        self.generated_text = []
        self.policy.eval()
        running_reward, idx_step = 0, 0
        for i_episode in range(num_episodes):
            for ref_question in self.env.ref_questions:
                state, ep_reward = self.env.reset(), 0
                for t in range(0, self.env.max_len):
                    action, log_probs, value, (valid_actions, actions_probs), dist = self.select_action(state=state,
                                                                                                        num_truncated=self.num_truncated)
                    idx_step += 1
                    state, (reward, closest_question), done, _ = self.env.step(action)
                    for metric in self.test_metrics:
                        metric.fill(state=state, done=done, dist=dist, valid_actions=valid_actions,
                                    ref_question=ref_question, reward=reward, closest_question=closest_question)
                    if done:
                        break
            for metric in self.test_metrics:
                metric.compute() #TODO: change reward here?
            if i_episode % log_interval == 0:
                for metric in self.test_metrics:
                    metric.write()
                # TODO: add generated dialog.
                # TODO: add ratio of unique closest question
                # TODO: add %age of unvalid actions per episode. (counter.)

    def learn(self, log_interval=10, num_episodes=100):

        running_reward = 0
        timestep = 0
        for i_episode in range(num_episodes):
            state, ep_reward = self.env.reset(), 0
            ref_question = random.choice(self.env.ref_questions)
            for t in range(0, self.env.max_len):
                forced = ref_question[t] if self.pretrain else None
                action, log_probs, value, (valid_actions, actions_probs), dist = self.select_action(state=state,
                                                                                                    forced=forced,
                                                                                                    num_truncated=self.num_truncated)
                state, (reward, closest_question), done, _ = self.env.step(action)
                # Saving reward and is_terminal:
                self.memory.rewards.append(reward)
                self.memory.is_terminals.append(done)

                timestep += 1
                for metric in self.train_metrics:
                    metric.fill(state=state, done=done, dist=dist, valid_actions=valid_actions,
                                actions_probs=actions_probs,
                                ref_question=self.env.ref_questions_decoded, reward=reward,
                                closest_question=closest_question)

                # update if its time
                if self.update_mode == "step" and timestep % self.update_every == 0:
                    self.update()
                    self.memory.clear_memory()
                    timestep = 0

                ep_reward += reward
                if done:
                    for metric in self.train_metrics:
                        metric.compute()
                        metric.reset()
                    self.writer.add_scalar('train_TTR', self.get_metrics(state.text), i_episode + 1)
                    if self.update_mode == "episode" and i_episode % self.update_every == 0:
                        self.update()
                        self.memory.clear_memory()
                    break

            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            if i_episode % log_interval == 0:
                logging.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                    i_episode, ep_reward, running_reward))
                logging.info('Episode questions: {}'.format(self.env.ref_questions_decoded))
                #self.writer.add_text('episode_questions', '  \n'.join(self.env.ref_questions_decoded))
                self.writer.add_scalar('train_running_return', running_reward, i_episode + 1)
                for metric in self.train_metrics:
                    metric.write()
