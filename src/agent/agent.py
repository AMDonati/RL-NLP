# TODO: add color logging:
# https://pypi.org/project/colorlog/
# https://medium.com/@galea/python-logging-example-with-color-formatting-file-handlers-6ee21d363184

import logging
import random
import torch
import torch.optim as optim
from eval.metric import metrics
from RL_toolbox.truncation import truncations
import time
import os
import numpy as np
from torch.distributions import Categorical
from utils.utils_train import write_to_csv


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
    def __init__(self, policy, env, writer, pretrained_lm, out_path, gamma=1., lr=1e-2, eps=1e-08, grad_clip=None,
                 lm_sl=True,
                 pretrain=False, update_every=50,
                 num_truncated=10, p_th=None, truncate_mode="top_k", log_interval=10, test_envs=[], eval_no_trunc=0):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.policy = policy
        self.policy.to(self.device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr, eps=eps)
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.log_interval = log_interval
        self.test_envs = test_envs
        self.pretrained_lm = pretrained_lm
        self.truncate_mode = truncate_mode
        if self.pretrained_lm is not None:
            self.pretrained_lm.to(self.device)
        self.lm_sl = lm_sl
        self.env = env
        self.pretrain = pretrain
        self.update_every = update_every
        self.memory = Memory()
        self.num_truncated = num_truncated
        self.eval_no_trunc = eval_no_trunc
        p_th_ = p_th if p_th is not None else 1 / self.env.clevr_dataset.len_vocab
        if truncate_mode is not None:
            self.truncation = truncations[truncate_mode](self, num_truncated=num_truncated, p_th=p_th_)  # adding the truncation class.
        else:
            self.truncation = truncations["no_trunc"](self, num_truncated=num_truncated, p_th=p_th_)
        self.writer = writer
        self.out_path = out_path
        self.checkpoints_path = os.path.join(out_path, "checkpoints")
        if not os.path.isdir(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        self.generated_text = []
        self.init_metrics()
        self.start_episode = 0

    def init_metrics(self):
        self.test_metrics = {key: metrics[key](self, train_test="test") for key in
                             ["reward", "dialog", "bleu", "ppl", "ppl_dialog_lm", "ttr_question", 'unique_words', 'ratio_closest_questions']}
        self.train_metrics = {key: metrics[key](self, train_test="train") for key in
                              ["lm_valid_actions", "policies_discrepancy", "valid_actions", "dialog"]}
        if self.truncate_mode == 'sample_va' or self.truncate_mode == 'proba_thr':
            self.train_metrics["size_valid_actions"] = metrics["size_valid_actions"](self, train_test="train")

    def select_action(self, state):
        valid_actions, action_probs = self.truncation.get_valid_actions(state)
        policy_dist, policy_dist_truncated, value = self.truncation.get_policy_distributions(state, valid_actions)
        action = self.truncation.sample_action(policy_dist=policy_dist, policy_dist_truncated=policy_dist_truncated,
                                               valid_actions=valid_actions)
        log_prob = policy_dist.log_prob(action.to(self.device)).view(-1)
        log_prob_truncated = policy_dist_truncated.log_prob(action.to(self.device)).view(-1)
        if self.policy.train_policy == 'truncated':
            assert torch.all(torch.eq(policy_dist_truncated.probs, policy_dist.probs))
        return action, log_prob, value, (valid_actions, action_probs, log_prob_truncated), policy_dist

    def generate_action_test(self, state, truncation=False, test_mode='sampling'):
        with torch.no_grad():
            if truncation:
                valid_actions, action_probs = self.truncation.get_valid_actions(state)
            else:
                valid_actions, action_probs = None, None
            policy_dist, policy_dist_truncated, value = self.policy(state.text, state.img, valid_actions)
            if test_mode == 'sampling':
                action = policy_dist_truncated.sample()
            elif test_mode == 'greedy':
                action = torch.argmax(policy_dist_truncated.probs).view(1).detach()  # TODO: remove the detach here?
            if policy_dist_truncated.probs.size() != policy_dist.probs.size():
                action = torch.gather(valid_actions, 1, action.view(1, 1))
            log_prob = policy_dist.log_prob(action.to(self.device)).view(-1)
            log_prob_truncated = policy_dist_truncated.log_prob(action.to(self.device)).view(-1)
        return action, log_prob, value, (valid_actions, action_probs, log_prob_truncated), policy_dist

    def generate_one_episode_test(self, env, truncation, test_mode, seed=None):
        state, ep_reward = env.reset(seed=seed), 0
        for t in range(0, env.max_len):
            action, log_probs, value, (
            valid_actions, action_probs, log_prob_truncated), dist = self.generate_action_test(state=state,
                                                                                               truncation=truncation,
                                                                                               test_mode=test_mode)
            new_state, (reward, closest_question), done, _ = env.step(action.cpu().numpy())
            ep_reward += reward
            for key, metric in self.test_metrics.items():
                if key != "ppl":
                    metric.fill(state=state, done=done, new_state=new_state,
                                ref_question=env.ref_questions, reward=reward,
                                closest_question=closest_question, dist=dist, valid_actions=valid_actions)
                else:
                    # computing ppl on ref questions only in one case (otherwise redundant).
                    if not truncation and test_mode == "sampling":
                        metric.fill(state=state, done=done, new_state=new_state,
                                    ref_question=env.ref_questions, reward=reward,
                                    closest_question=closest_question,
                                    dist=dist, valid_actions=valid_actions)
            state = new_state
            if done:
                break
        for key, metric in self.test_metrics.items():
            if key != "ppl":
                metric.compute(state=state, closest_question=closest_question,
                               reward=reward, img_idx=env.img_idx, ref_question=env.ref_questions)
                metric.write()
            else:
                if not truncation and test_mode == "sampling":
                    metric.compute(state=state, closest_question=closest_question,
                                   reward=reward, img_idx=env.img_idx, ref_question=env.ref_questions)
                    metric.write()
        return state, ep_reward, closest_question, env.img_idx


    def generate_one_episode_with_lm(self, env, test_mode='sampling'):
        state = env.special_tokens.SOS_idx
        state = torch.LongTensor([state]).view(1, 1).to(self.device)
        with torch.no_grad():
            for i in range(env.max_len):
                log_probas, hidden = self.pretrained_lm(state)  # output (1, num_tokens)
                if test_mode == 'sampling':
                    softmax = log_probas[-1, :].squeeze().exp()
                    word_idx = Categorical(softmax).sample()
                elif test_mode == 'greedy':
                    word_idx = log_probas[-1, :].squeeze().argmax()
                state = torch.cat([state, word_idx.view(1, 1)], dim=1)
                if word_idx == env.special_tokens.EOS_idx:
                    break
        new_state = env.State(state, None)  # trick to have a state.text in the metric.
        state_decoded = self.env.clevr_dataset.idx2word(state.squeeze().cpu().numpy(), stop_at_end=True, ignored=['<SOS>'])
        # compute associated reward with reward function:
        reward, closest_question = env.reward_func.get(question=state_decoded,
                                                       ep_questions_decoded=env.ref_questions_decoded,
                                                       step_idx=env.max_len, done=True)

        for key in ["reward", "ppl_dialog_lm"]:
            self.test_metrics[key].reinit_train_test(self.test_metrics[key].train_test + '_' + 'fromLM')
            self.test_metrics[key].fill(done=True, new_state=new_state, reward=reward,
                                        closest_question=closest_question)  # TODO: does not work with differential reward.
            self.test_metrics[key].compute()
        # reset metrics key value for writing:
        for m in self.test_metrics.values():
            m.reinit_train_test(env.mode + '_' + test_mode)
        return state, state_decoded, reward, closest_question


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


    def test(self, num_episodes=10, test_mode='sampling'):
        for env in self.test_envs:
            logging.info('-----------------------Starting Evaluation for {} dialog ------------------'.format(env.mode))
            self.test_env(env, num_episodes=num_episodes, test_mode=test_mode)


    def test_env(self, env, num_episodes=10, test_mode='sampling'):
        # init env:
        env.reset()
        for m in self.test_metrics.values():
            m.reinit_train_test(env.mode + '_' + test_mode)
        self.generated_text = []
        self.policy.eval()
        if self.eval_no_trunc == 1:
            # if using truncation, eval the test dialog with and without truncation
            truncation = {"no_trunc": False, "with_trunc": True} if self.truncate_mode is not None else {"no_trunc": False}
        else:
            # if using truncation, eval the test dialog only with truncation
            truncation = {"with_trunc": True} if self.truncate_mode is not None else {
                "no_trunc": False}
        for i_episode in range(num_episodes):
            dialogs = {key:[] for key in truncation.keys()}
            logging.info(
                '-------------Test Episode: {} --------------------------------------------------------------------------------------'.format(
                    i_episode))
            seed = np.random.randint(1000000)  # setting the seed to generate the episode with the same image.
            for key, trunc in truncation.items():
                for m in self.test_metrics.values():
                    m.reinit_train_test(m.train_test + '_' + key)
                for i in range(env.ref_questions.size(0)): # loop multiple time over the same image to measure langage diversity
                    state, ep_reward, closest_question, img_idx = self.generate_one_episode_test(env=env, truncation=trunc,
                                                                                                      test_mode=test_mode,
                                                                                                      seed=seed)
                    dialogs[key].append('DIALOG {} for img {}: {}:'.format(i, img_idx, key) + self.env.clevr_dataset.idx2word(state.text[:, 1:].numpy()[
                                                                                                  0]) + '----- closest question:' + closest_question + '------reward: {}'.format(
                        ep_reward))
                    if i == env.ref_questions.size(0) - 1:
                        # reset metrics key value for writing:
                        for m in self.test_metrics.values():
                            m.reinit_train_test(env.mode + '_' + test_mode)
            # generate one question with the lm as a comparison
            _, dialog_from_lm, ep_reward, closest_question = self.generate_one_episode_with_lm(env=env, test_mode=test_mode)
            dialogs["from_lm"] = ['DIALOG from Language Model: {}'.format(dialog_from_lm) + '----- closest question:' + closest_question + '------reward: {}'.format(
                            ep_reward)]
            for _, dialog in dialogs.items():
                logging.info('\n'.join(dialog))
            logging.info(
                '-------------------------------------------------------------------------------------------------------------------------------------------------------')


    def learn(self, num_episodes=100):
        start_time = time.time()
        current_time = time.time()
        running_reward = 0
        timestep = 1
        self.dict_running_return = {}
        for i_episode in range(self.start_episode, self.start_episode + num_episodes):
            state, ep_reward = self.env.reset(), 0
            # ref_question = random.choice(self.env.ref_questions)
            ep_log_probs, ep_log_probs_truncated, lm_log_probs = [], [], []  # TODO: use the Memory Class or the Metric Class instead.
            for t in range(0, self.env.max_len):
                # forced = ref_question[t] if self.pretrain else None
                action, log_probs, value, (
                    valid_actions, actions_probs, log_probs_truncated), dist = self.select_action(state=state)
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
                        metric.compute(state=state, closest_question=closest_question, img_idx=self.env.img_idx)
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
            self.dict_running_return[i_episode] = running_reward
            if i_episode % self.log_interval == 0:
                logging.info(
                    "----------------------------------------- Episode {} - Img  {} -------------------------------------------------------".format(
                        i_episode, self.env.img_idx))
                logging.info('Last reward: {:.2f}\tAverage reward: {:.2f}'.format(ep_reward, running_reward))
                # logging.info('Episode questions: {}'.format(self.env.ref_questions_decoded))
                logging.info('LAST DIALOG: {}'.format(self.env.clevr_dataset.idx2word(state.text[:, 1:].numpy()[0])))
                logging.info('Closest Question: {}'.format(closest_question))
                logging.info('episode action probs: {}'.format(ep_probs))
                if valid_actions is not None:
                    logging.info('episode action probs truncated: {}'.format(
                        ep_probs_truncated))  # to monitor the discrepancy between the truncated softmax and the softmax over the whole action space.
                    logging.info('episode action probs from the LANGUAGE MODEL: {}'.format(ep_lm_probs))
                    logging.info('---------------------Valid action space------------------------------')
                    logging.info('\n'.join(self.train_metrics["valid_actions"].metric))
                logging.info(
                    "---------------------------------------------------------------------------------------------------------------------------------------")
                self.writer.add_scalar('train_running_return', running_reward, i_episode + 1)
                self.writer.add_scalar("train_action_probs", np.mean(ep_probs), i_episode + 1)
                self.writer.add_scalar("train_action_probs_truncated", np.mean(ep_probs_truncated), i_episode + 1)
                if valid_actions is not None:
                    self.writer.add_scalar("train_action_probs_lm", np.mean(ep_lm_probs), i_episode + 1)
                for key, metric in self.train_metrics.items():
                    if key != "valid_actions":  # not taking the valid_actions metric. Used only in logging.
                        metric.write()

            if i_episode + 1 % 1000 == 0:
                elapsed = time.time() - current_time
                logging.info("Training time for 1000 episodes: {:5.2f}".format(elapsed))
                current_time = time.time()
                # saving checkpoint:
                self.save_ckpt(EPOCH=i_episode, loss=loss)
        if valid_actions is not None:  # to compare the discrepancy between the 'truncated policy' and the 'all space' policy that is being learned. (Plus comparison with the language model).
            self.writer.add_custom_scalars({'Train_all_probs': {'action_probs': ['Multiline', ['train_action_probs',
                                                                                               'train_action_probs_truncated',
                                                                                               'train_action_probs_lm']]}})
        # writing to csv the history of running return:
        write_to_csv(os.path.join(self.out_path, "train_running_return_history.csv"),
                     self.dict_running_return)  # useful to have a metric to monitor the convergence speed.
        # write to csv other metric:
        for _, metric in self.train_metrics.items():
            metric.write_to_csv()
        logging.info("total training time: {:7.2f}".format(time.time() - start_time))
        logging.info("running_reward: {}".format(running_reward))
        logging.info(
            "--------------------------------------------END OF TRAINING ----------------------------------------------------")
