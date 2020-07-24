# TODO: add color logging:
# https://pypi.org/project/colorlog/
# https://medium.com/@galea/python-logging-example-with-color-formatting-file-handlers-6ee21d363184

import logging
import random
import torch
import torch.optim as optim
from nltk.translate.bleu_score import sentence_bleu
from eval.metric import metrics
import time
import os
import numpy as np


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
        self.arrs = [self.actions, self.states_text, self.states_img, self.logprobs, self.rewards,
                     self.is_terminals, self.values]

        self.idx_episode = 0

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.states_img[:]
        del self.states_text[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

    def add_step(self, actions, states_text, states_img, logprobs, rewards, is_terminals, values):
        for arr, val in zip(self.arrs, [actions, states_text, states_img, logprobs, rewards, is_terminals, values]):
            arr.append(val)


class Agent:
    def __init__(self, policy, env, writer, out_path, gamma=1., lr=1e-2, eps=1e-08, grad_clip=None, pretrained_lm=None,
                 lm_sl=True,
                 pretrain=False, update_every=50,
                 num_truncated=10, log_interval=10, test_envs=[]):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=eps)
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.log_interval = log_interval
        self.test_envs = test_envs
        self.pretrained_lm = pretrained_lm
        if self.pretrained_lm is not None:
            self.pretrained_lm.to(self.device)
        self.lm_sl = lm_sl
        self.env = env
        self.pretrain = pretrain
        self.update_every = update_every
        self.memory = Memory()
        self.num_truncated = num_truncated
        self.writer = writer
        self.checkpoints_path = os.path.join(out_path, "checkpoints")
        if not os.path.isdir(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        self.generated_text = []
        self.init_metrics()
        self.start_episode = 0

    def init_metrics(self):
        self.test_metrics = {key: metrics[key](self, train_test="test") for key in ["reward", "dialog"]}
        self.train_metrics = {key: metrics[key](self, train_test="train") for key in
                              ["lm_valid_actions", "policies_discrepancy", "lm_policy_probs_ratio", "valid_actions"]}

    def get_top_k_words(self, state_text, top_k=10, state_img=None):
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
            log_probas, _ = self.pretrained_lm(state_text.to(self.device))
            log_probas = log_probas.view(len(state_text), seq_len, -1)
            log_probas = log_probas[:, -1, :]
            top_k_weights, top_k_indices = torch.topk(log_probas, top_k, sorted=True)
        else:
            dist, dist_, value = self.pretrained_lm(state_text, state_img)
            probs = dist.probs
            top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)

        return top_k_indices, top_k_weights

    def select_action(self, state, forced=None, num_truncated=10):
        pass

    def finish_episode(self):
        pass

    def save(self, out_file):
        with open(out_file, 'wb') as f:
            torch.save(self.policy.state_dict(), f)

    def save_ckpt(self, EPOCH, loss):
        torch.save({
            'epoch': EPOCH,
            'model_state_dict': self.policy.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
        }, os.path.join(self.checkpoints_path, 'model.pt'))

    def load_ckpt(self):
        checkpoint = torch.load(os.path.join(self.checkpoints_path, 'model.pt'))
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']

        return epoch, loss

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
        for env in self.test_envs:
            self.test_env(env, log_interval=log_interval, num_episodes=num_episodes)

    def test_env(self, env, log_interval=1, num_episodes=10):
        for m in self.test_metrics.values():
            m.train_test = env.mode
        self.generated_text = []
        self.policy.eval()
        running_reward, idx_step = 0, 0
        for i_episode in range(num_episodes):
            state, ep_reward = env.reset(), 0
            for t in range(0, env.max_len):
                action, log_probs, value, (
                    valid_actions, actions_probs, log_probs_truncated), dist = self.select_action(state=state,
                                                                                                  num_truncated=self.num_truncated)
                idx_step += 1
                new_state, (reward, closest_question), done, _ = env.step(action.cpu().numpy())
                for key, metric in self.test_metrics.items():
                    metric.fill(state=state, done=done, dist=dist, valid_actions=valid_actions,
                                ref_question=env.ref_questions_decoded, reward=reward,
                                closest_question=closest_question, new_state=new_state, log_probs=log_probs,
                                log_probs_truncated=log_probs_truncated)
                state = new_state

                if done:
                    break
            for key, metric in self.test_metrics.items():
                metric.compute(state=state, closest_question=closest_question,
                               reward=reward)
            if i_episode % log_interval == 0:
                for key, metric in self.test_metrics.items():
                    metric.write()

            # TODO: add the mean's reward and variance.

    def learn(self, num_episodes=100):
        start_time = time.time()
        current_time = time.time()
        running_reward = 0
        timestep = 1
        for i_episode in range(self.start_episode, self.start_episode + num_episodes):
            state, ep_reward = self.env.reset(), 0
            ref_question = random.choice(self.env.ref_questions)
            ep_log_probs, ep_log_probs_truncated, lm_log_probs = [], [], []  # TODO: use a class or a dictionnary instead.
            for t in range(0, self.env.max_len):
                forced = ref_question[t] if self.pretrain else None
                action, log_probs, value, (
                    valid_actions, actions_probs, log_probs_truncated), dist = self.select_action(state=state,
                                                                                                  forced=forced,
                                                                                                  num_truncated=self.num_truncated)
                ep_log_probs.append(log_probs)
                ep_log_probs_truncated.append(log_probs_truncated)
                if valid_actions is not None:
                    lm_log_probs.append(actions_probs[valid_actions == action])
                new_state, (reward, closest_question), done, _ = self.env.step(action.cpu().numpy())
                # Saving reward and is_terminal:
                self.memory.add_step(action, state.text[0], state.img[0], log_probs, reward, done, value)

                timestep += 1
                for key, metric in self.train_metrics.items():
                    metric.fill(state=state, action=action, done=done, dist=dist, valid_actions=valid_actions,
                                actions_probs=actions_probs,
                                ref_question=self.env.ref_questions_decoded, reward=reward,
                                closest_question=closest_question, new_state=new_state, log_probs=log_probs,
                                log_probs_truncated=log_probs_truncated)
                state = new_state

                # update if its time
                if self.update_mode == "step" and timestep % self.update_every == 0:
                    loss = self.update()
                    logging.info("UPDATING POLICY WEIGHTS...")
                    self.memory.clear_memory()
                    timestep = 0

                ep_reward += reward
                if done:
                    for key, metric in self.train_metrics.items():
                        metric.compute()
                    if self.update_mode == "episode" and i_episode % self.update_every == 0:
                        loss = self.update()
                        logging.info("UPDATING POLICY WEIGHTS...")
                        self.memory.clear_memory()
                    break
            ep_log_probs = torch.stack(ep_log_probs).clone().detach()
            ep_probs = np.round(np.exp(ep_log_probs.cpu().squeeze().numpy()), decimals=5)
            ep_log_probs_truncated = torch.stack(ep_log_probs_truncated).clone().detach()
            ep_probs_truncated = np.round(np.exp(ep_log_probs_truncated.cpu().squeeze().numpy()), decimals=5)
            if valid_actions is not None:
                lm_log_probs = torch.stack(lm_log_probs).clone().detach()
                ep_lm_probs = np.round(np.exp(lm_log_probs.cpu().squeeze().numpy()), decimals=5)
            running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
            if i_episode % self.log_interval == 0:
                logging.info(
                    "----------------------------------------- Episode {} -------------------------------------------------------".format(
                        i_episode))
                logging.info('Last reward: {:.2f}\tAverage reward: {:.2f}'.format(ep_reward, running_reward))
                # logging.info('Episode questions: {}'.format(self.env.ref_questions_decoded))
                logging.info('LAST DIALOG: {}'.format(self.env.clevr_dataset.idx2word(state.text[:, 1:].numpy()[0])))
                logging.info('Closest Question: {}'.format(closest_question))
                logging.info('episode action probs: {}'.format(ep_probs))
                if valid_actions is not None:
                    logging.info('episode action probs truncated: {}'.format(ep_probs_truncated))
                    logging.info('episode action probs from the LANGUAGE MODEL: {}'.format(ep_lm_probs))
                    logging.info('---------------------Valid action space------------------------------')
                    logging.info('\n'.join(self.train_metrics["valid_actions"].metric))
                logging.info(
                    "---------------------------------------------------------------------------------------------------------------------------------------")
                self.writer.add_scalar('train_running_return', running_reward, i_episode + 1)
                self.writer.add_scalar("train_action_probs", np.mean(ep_probs), i_episode + 1)
                self.writer.add_scalar("train_action_probs_truncated", np.mean(ep_probs_truncated), i_episode + 1)
                self.writer.add_scalar("train_action_probs_lm", np.mean(ep_lm_probs), i_episode + 1)
                for key, metric in self.train_metrics.items():
                    if key != 'valid_actions':  # not taking the valid_actions metric.
                        metric.write()

            if i_episode + 1 % 1000 == 0:
                elapsed = time.time() - current_time
                logging.info("Training time for 1000 episodes: {:5.2f}".format(elapsed))
                current_time = time.time()
                # saving checkpoint:
                self.save_ckpt(EPOCH=i_episode, loss=loss)
        if self.pretrained_lm is not None:
            self.writer.add_custom_scalars({'Train_all_probs': {'action_probs': ['Multiline', ['train_action_probs',
                                                                                       'train_action_probs_truncated',
                                                                                       'train_action_probs_lm']]}})
        logging.info(
            "--------------------------------------------END OF TRAINING ----------------------------------------------------")
        logging.info("total training time: {:7.2f}".format(time.time() - start_time))
        logging.info("running_reward: {}".format(running_reward))
