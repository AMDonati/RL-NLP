# TODO: add color logging:
# https://pypi.org/project/colorlog/
# https://medium.com/@galea/python-logging-example-with-color-formatting-file-handlers-6ee21d363184

import logging
import os
import time

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from RL_toolbox.truncation import truncations
from agent.memory import Memory
from eval.metric import metrics


class Agent:
    def __init__(self, policy, env, writer, pretrained_lm, out_path, gamma=1., lr=1e-2, grad_clip=None,
                 pretrain=False, update_every=50,
                 num_truncated=10, p_th=None, truncate_mode="top_k", log_interval=10, test_envs=[], eval_no_trunc=0,
                 alpha_logits=1., alpha_decay_rate=0.):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy.to(self.device)
        self.start_policy = policy  # keep pretrained policy (or random policy if not pretrain) as a test baseline.
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.log_interval = log_interval
        self.test_envs = test_envs
        self.pretrained_lm = pretrained_lm
        self.truncate_mode = truncate_mode
        if self.pretrained_lm is not None:
            self.pretrained_lm.to(self.device)
        self.alpha_logits_lm = alpha_logits
        self.alpha_decay_rate = alpha_decay_rate
        self.env = env
        self.pretrain = pretrain
        self.update_every = update_every
        self.memory = Memory()
        self.num_truncated = num_truncated
        if self.truncate_mode is not None:
            self.eval_trunc = {"no_trunc": False, "with_trunc": True} if eval_no_trunc else {"with_trunc": True}
        else:
            self.eval_trunc = {"no_trunc": False}
        p_th_ = p_th if p_th is not None else 1 / self.env.clevr_dataset.len_vocab
        if truncate_mode is not None:
            self.truncation = truncations[truncate_mode](self, num_truncated=num_truncated,
                                                         p_th=p_th_)  # adding the truncation class.
        else:
            self.truncation = truncations["no_trunc"](self, num_truncated=num_truncated, p_th=p_th_)
        self.writer = writer
        self.out_path = out_path
        self.checkpoints_path = os.path.join(out_path, "checkpoints")
        if not os.path.isdir(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        self.generated_text = []
        self.init_metrics()
        self.start_episode = 1

    def init_metrics(self):
        self.test_metrics = {key: metrics[key](self, train_test="test") for key in
                             ["reward", "dialog", "bleu", "ppl", "ppl_dialog_lm", "ttr_question", 'unique_words']}
        if self.env.reward_type == 'levenshtein_':
            self.test_metrics['ratio_closest_questions'] = metrics["ratio_closest_questions"](self, train_test="test")
        self.train_metrics = {key: metrics[key](self, train_test="train") for key in
                              ["running_return", "lm_valid_actions", "policies_discrepancy", "valid_actions", "dialog",
                               "policy", "action_probs", "action_probs_truncated"]}
        if self.truncate_mode is not None:
            for key in ["action_probs_lm"]:
                self.train_metrics[key] = metrics[key](self, train_test="train")
        if self.truncate_mode == 'sample_va' or self.truncate_mode == 'proba_thr':
            self.train_metrics["size_valid_actions"] = metrics["size_valid_actions"](self, train_test="train")

    def decay_alpha_logits_lm(self, i_episode, alpha_min=0.001, update_every=500):
        if self.alpha_decay_rate > 0 and self.alpha_logits_lm > alpha_min:
            if i_episode % update_every == 0:
                self.alpha_logits_lm *= (1-self.alpha_decay_rate)
                logging.info("decaying alpha logits parameter at Episode #{} - new value: {}".format(i_episode,
                                                                                                     self.alpha_logits_lm))

    def select_action(self, state, mode='sampling', truncation=True, baseline=False, train=True):
        valid_actions, action_probs, logits_lm = self.truncation.get_valid_actions(state, truncation)
        alpha = self.alpha_logits_lm if train else 0
        policy_dist, policy_dist_truncated, value = self.truncation.get_policy_distributions(state, valid_actions,
                                                                                             logits_lm,
                                                                                             baseline=baseline,
                                                                                             alpha=alpha)
        action = self.truncation.sample_action(policy_dist=policy_dist, policy_dist_truncated=policy_dist_truncated,
                                               valid_actions=valid_actions,
                                               mode=mode)
        log_prob = policy_dist.log_prob(action.to(self.device)).view(-1)
        log_prob_truncated = policy_dist_truncated.log_prob(action.to(self.device)).view(-1)
        if self.policy.train_policy == 'truncated':
            assert torch.all(torch.eq(policy_dist_truncated.probs, policy_dist.probs))
        return action, log_prob, value, (valid_actions, action_probs, log_prob_truncated), policy_dist

    def generate_one_episode_with_lm(self, env, test_mode='sampling'):  # TODO: refactor this with alpha = 1.
        state, ep_reward = env.reset(), 0
        with torch.no_grad():
            for i in range(env.max_len):
                log_probas, hidden = self.pretrained_lm(state.text)  # output (1, num_tokens)
                if test_mode == 'sampling':
                    softmax = log_probas[-1, :].squeeze().exp()
                    action = Categorical(softmax).sample()
                elif test_mode == 'greedy':
                    action = log_probas[-1, :].squeeze().argmax()
                new_state, (reward, closest_question, pred_answer), done, _ = env.step(action.cpu().numpy())
                state = new_state
                ep_reward += reward
                if done:
                    break
        for key in ["reward", "ppl_dialog_lm", "bleu"]:
            self.test_metrics[key].reinit_train_test(self.test_metrics[key].train_test + '_' + 'fromLM')
            self.test_metrics[key].fill(done=True, new_state=new_state, reward=reward,
                                        closest_question=closest_question,
                                        state=state,
                                        ref_questions_decoded=env.ref_questions_decoded)
            self.test_metrics[key].compute()
        # reset metrics key value for writing:
        for m in self.test_metrics.values():
            m.reinit_train_test(env.mode + '_' + test_mode)
        return state, env.dialog, ep_reward, closest_question

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

    def load_ckpt(self, ckpt_path):
        checkpoint = torch.load(os.path.join(ckpt_path, 'model.pt'))
        self.policy.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        return epoch, loss

    def test(self, num_episodes=10, test_mode='sampling', baselines=False):
        for env in self.test_envs:
            logging.info('-----------------------Starting Evaluation for {} dialog ------------------'.format(env.mode))
            self.test_env(env, num_episodes=num_episodes, test_mode=test_mode, baselines=baselines)

    def generate_one_episode(self, timestep, i_episode, env, seed=None, train=True, truncation=True,
                             test_mode='sampling', baseline=False):
        state, ep_reward = env.reset(seed), 0
        metrics = self.train_metrics if train else self.test_metrics
        for t in range(0, env.max_len):
            action, log_probs, value, (
                valid_actions, actions_probs, log_probs_truncated), dist = self.select_action(state=state,
                                                                                              train=train,
                                                                                              mode=test_mode,
                                                                                              truncation=truncation,
                                                                                              baseline=baseline)
            new_state, (reward, closest_question, pred_answer), done, _ = env.step(action.cpu().numpy())
            if train:
                # Saving reward and is_terminal:
                self.memory.add_step(action, state.text[0], state.img[0], log_probs, reward, done, value,
                                     state.answer)
            timestep += 1
            for key, metric in metrics.items():
                metric.fill(state=state, action=action, done=done, dist=dist, valid_actions=valid_actions,
                            actions_probs=actions_probs,
                            ref_question=env.ref_questions,
                            ref_questions_decoded=env.ref_questions_decoded, reward=reward,
                            closest_question=closest_question, new_state=new_state, log_probs=log_probs,
                            log_probs_truncated=log_probs_truncated,
                            test_mode=test_mode,
                            pred_answer=pred_answer)
            state = new_state
            ep_reward += reward

            # update if its time
            if train:
                if self.update_mode == "step" and timestep % self.update_every == 0:
                    loss = self.update()
                    logging.info("UPDATING POLICY WEIGHTS...")
                    self.memory.clear_memory()
                    timestep = 0
                else:
                    loss = None

            if done:
                if train:
                    if self.update_mode == "episode" and i_episode % self.update_every == 0:
                        loss = self.update()
                        logging.info("UPDATING POLICY WEIGHTS...")
                        self.memory.clear_memory()
                else:
                    loss = None
                break
        for key, metric in metrics.items():
            metric.compute(state=state, closest_question=closest_question, img_idx=env.img_idx, reward=reward,
                           ref_question=env.ref_questions, test_mode=test_mode, pred_answer=pred_answer)

        return state, ep_reward, closest_question, valid_actions, timestep, loss

    def test_env(self, env, num_episodes=10, test_mode='sampling', baselines=False):
        env.reset()  # init env.
        timestep = 1
        for m in self.test_metrics.values():
            m.reinit_train_test(env.mode + '_' + test_mode)
        self.policy.eval()
        for i_episode in range(num_episodes):
            dialogs = {key: [] for key in self.eval_trunc.keys()}
            logging.info('-' * 20 + 'Test Episode: {}'.format(i_episode) + '-' * 20)
            seed = np.random.randint(1000000)  # setting the seed to generate the episode with the same image.
            for key, trunc in self.eval_trunc.items():
                for m in self.test_metrics.values():
                    m.reinit_train_test(m.train_test + '_' + key)
                for i in range(env.ref_questions.size(
                        0)):  # loop multiple time over the same image to measure langage diversity
                    with torch.no_grad():
                        state, ep_reward, closest_question, valid_actions, timestep, _ = self.generate_one_episode(
                            timestep=timestep, i_episode=i_episode, env=env, seed=seed, train=False,
                            test_mode=test_mode,
                            truncation=trunc)
                    for _, metric in self.test_metrics.items():
                        metric.write()
                    dialogs[key].append(
                        'DIALOG {} for img {} - question_index {} - {}: '.format(i, env.img_idx, env.data_idx, key) + self.env.clevr_dataset.idx2word(
                            state.text[:, 1:].numpy()[
                                0]) + '----- closest question:' + closest_question + '------ reward: {}'.format(
                            ep_reward))
                    if i == env.ref_questions.size(0) - 1:
                        # reset metrics key value for writing:
                        for m in self.test_metrics.values():
                            m.reinit_train_test(env.mode + '_' + test_mode)
            if baselines:
                # generate one question with the lm as a comparison
                _, dialog_from_lm, ep_reward, closest_question = self.generate_one_episode_with_lm(env=env,
                                                                                                   test_mode=test_mode)
                dialogs["from_lm"] = ['DIALOG from Language Model: {}'.format(
                    dialog_from_lm) + '----- closest question:' + closest_question + '------reward: {}'.format(
                    ep_reward)]
                # generate one question with the start policy as a comparison
                for m in self.test_metrics.values():
                    m.reinit_train_test(env.mode + '_' + test_mode + '_' + 'StartPolicy')
                with torch.no_grad():
                    state, ep_reward, closest_question, valid_actions, timestep, _ = self.generate_one_episode(
                        timestep=timestep, i_episode=i_episode, env=env, seed=seed, train=False, test_mode=test_mode,
                        truncation=False, baseline=True)
                dialogs["start_policy"] = ['DIALOG from start policy: {}'.format(self.env.clevr_dataset.idx2word(
                    state.text[:, 1:].numpy()[
                        0])) + '----- closest question:' + closest_question + '------reward: {}'.format(
                    ep_reward)]
                for m in self.test_metrics.values():
                    m.reinit_train_test(env.mode + '_' + test_mode)
            for _, dialog in dialogs.items():
                logging.info('\n'.join(dialog))
            logging.info(
                '-------------------------------------------------------------------------------------------------------------------------------------------------------')

    def learn(self, num_episodes=100):
        start_time = time.time()
        current_time = time.time()
        timestep = 1
        for i_episode in range(self.start_episode, self.start_episode + num_episodes):
            state, ep_reward, closest_question, valid_actions, timestep, loss = self.generate_one_episode(
                timestep=timestep, i_episode=i_episode, env=self.env)
            self.decay_alpha_logits_lm(i_episode=i_episode)
            if i_episode % self.log_interval == 0:
                logging.info(
                    "----------------------------------------- Episode {} - Img  {} -------------------------------------------------------".format(
                        i_episode, self.env.img_idx))
                logging.info('Last reward: {:.2f}\tAverage reward: {:.2f}'.format(ep_reward, self.train_metrics[
                    "running_return"].metric[0]))
                # logging.info('Episode questions: {}'.format(self.env.ref_questions_decoded))
                logging.info('LAST DIALOG: {}'.format(self.env.clevr_dataset.idx2word(state.text[:, 1:].numpy()[0])))
                logging.info('Closest Question: {}'.format(closest_question))
                for key, metric in self.train_metrics.items():
                    metric.log(valid_actions=valid_actions)
                    metric.write()
                logging.info(
                    "---------------------------------------------------------------------------------------------------------------------------------------")

            if i_episode + 1 % 1000 == 0:
                elapsed = time.time() - current_time
                logging.info("Training time for 1000 episodes: {:5.2f}".format(elapsed))
                current_time = time.time()
                # saving checkpoint:
                self.save_ckpt(EPOCH=i_episode, loss=loss)
        if valid_actions is not None:  # to compare the discrepancy between the 'truncated policy' and the 'all space' policy
            self.writer.add_custom_scalars({'Train_all_probs': {'action_probs': ['Multiline', ['train_action_probs',
                                                                                               'train_action_probs_truncated',
                                                                                               'train_action_probs_lm']]}})
        # write to csv train metrics:
        for _, metric in self.train_metrics.items():
            metric.write_to_csv()
        logging.info("total training time: {:7.2f}".format(time.time() - start_time))
        logging.info(
            "--------------------------------------------END OF TRAINING ----------------------------------------------------")
