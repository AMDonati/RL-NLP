import logging
import os

os.environ['TRANSFORMERS_CACHE'] = "/cache"
import random
import time

import numpy as np
import torch
import pandas as pd

from RL_toolbox.truncation import truncations
from agent.memory import Memory
from eval.metric import metrics

logger = logging.getLogger()


class Agent:
    def __init__(self, policy, optimizer, env, writer, pretrained_lm, out_path, gamma=1., lr=1e-2, grad_clip=None,
                 scheduler=None,
                 pretrain=False, update_every=50,
                 num_truncated=10, p_th=None, truncate_mode="top_k", log_interval=10, test_envs=[], eval_no_trunc=0,
                 alpha_logits=0., alpha_decay_rate=0., epsilon_truncated=0., train_seed=0, epsilon_truncated_rate=1.,
                 is_loss_correction=1, train_metrics=[], test_metrics=[], top_p=1., temperature=1, temp_factor=1,
                 temperature_step=1, temperature_min=1, temperature_max=10, s_min=10, s_max=200, inv_schedule_step=0,
                 schedule_start=1, curriculum=0, KL_coeff=0., truncation_optim=0):
        self.device = policy.device
        self.policy = policy.to(self.device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.grad_clip = grad_clip
        self.gamma = gamma
        self.log_interval = log_interval
        self.test_envs = test_envs
        self.truncate_mode = truncate_mode
        self.alpha_logits_lm = alpha_logits
        self.alpha_decay_rate = alpha_decay_rate
        self.temperature = temperature
        self.temp_factor = temp_factor
        self.temperature_step = temperature_step
        self.temperature_min = temperature_min
        self.temperature_max = temperature_max
        self.inv_schedule_step = inv_schedule_step
        self.schedule_start = schedule_start
        self.env = env
        self.pretrain = pretrain
        self.update_every = update_every
        self.memory = Memory()
        self.num_truncated = num_truncated
        self.epsilon_truncated = epsilon_truncated
        self.epsilon_truncated_rate = epsilon_truncated_rate
        self.is_loss_correction = is_loss_correction
        self.curriculum = curriculum
        self.KL_coeff = KL_coeff
        self.truncation_optim = truncation_optim
        if self.curriculum > 0:
            self.env.update_mode(mode=env.mode, answer_sampl="random")
        p_th_ = p_th if p_th is not None else 1 / self.env.dataset.len_vocab

        if self.truncate_mode is not None:
            self.eval_trunc = {"no_trunc": False, "with_trunc": True} if eval_no_trunc else {"with_trunc": True}
            self.truncation = truncations[truncate_mode](self, num_truncated=num_truncated,
                                                         p_th=p_th_, pretrained_lm=pretrained_lm,
                                                         top_p=top_p, s_min=s_min,
                                                         s_max=s_max)  # adding the truncation class.
        else:
            self.eval_trunc = {"no_trunc": False}
            self.truncation = truncations["no_trunc"](self, num_truncated=num_truncated, p_th=p_th_, top_p=top_p,
                                                      pretrained_lm=pretrained_lm)

        self.writer = writer
        self.out_path = out_path
        self.checkpoints_path = os.path.join(out_path, "checkpoints")
        if not os.path.isdir(self.checkpoints_path):
            os.makedirs(self.checkpoints_path)
        self.generated_text = []
        self.train_metrics_names = train_metrics
        self.test_metrics_names = test_metrics
        self.init_metrics()
        self.start_episode = 1
        self.train_seed = train_seed
        if self.env.answer_sampling == "inv_frequency":
            inv_freq_answer_decoded = self.env.decode_inv_frequency()
            logger.info(
                "---------------- INV FREQ ANSWERS DISTRIBUTION FOR ANSWER SAMPLING--------------------------------")
            logger.info(inv_freq_answer_decoded)
            logger.info("-" * 100)
        if self.env.answer_sampling == "img_sampling":
            logger.info(
                "---------------- ANSWER / IMG STATS ---------------------------------------------------------------")
            min, mean, max = self.env.dataset.get_answer_img_stats()
            logger.info("number MIN of answers per img:{}".format(min))
            logger.info("number MEAN of answers per img:{}".format(mean))
            logger.info("number MAX of answers per img:{}".format(max))
            logger.info("-" * 100)

    def init_metrics(self):
        self.metrics = {}
        self.metrics["train"] = {
            key: metrics[key](self, train_test="train", env_mode="train", trunc="trunc",
                              sampling="sampling") for key in
            self.train_metrics_names if key in metrics}
        for mode in [env_.mode for env_ in self.test_envs]:
            for trunc in self.eval_trunc.keys():
                for sampling_mode in ["sampling", "greedy", "sampling_ranking_lm"]:
                    id = "_".join([mode, trunc, sampling_mode])
                    self.metrics[id] = {key: metrics[key](self, train_test="test", env_mode=mode, trunc=trunc,
                                                          sampling=sampling_mode) for key in self.test_metrics_names if
                                        key in metrics}

    def get_score_metric(self, metrics):
        # if self.truncation.language_model.__class__ == ClevrLanguageModel:
        #     if self.env.dataset.__class__ == CLEVR_Dataset:
        #         if self.truncation.language_model.lm_path == "output/lm_model/model.pt":
        #             score_metric = metrics["ppl_dialog_lm"]
        #         elif self.truncation.language_model.lm_path == "output/lm_ext/model.pt":
        #             score_metric = metrics["ppl_dialog_lm_ext"]
        #     else:
        #         score_metric = metrics["ppl_dialog_lm"]
        # else:
        score_metric = metrics["language_score"]
        return score_metric

    def get_metrics(self, mode, trunc, sampling_mode):
        id = "{}_{}_{}".format(mode, trunc, sampling_mode)
        return self.metrics[id]

    def update_per_episode(self, i_episode, alpha_min=0.001, update_every=500, num_episodes_train=1000):
        if self.alpha_decay_rate > 0 and self.alpha_logits_lm > alpha_min:
            if i_episode % update_every == 0:
                self.alpha_logits_lm *= (1 - self.alpha_decay_rate)
                logger.info("decaying alpha logits parameter at Episode #{} - new value: {}".format(i_episode,
                                                                                                    self.alpha_logits_lm))
        # if i_episode == int(self.epsilon_truncated_rate * num_episodes_train) + 1:
        # self.epsilon_truncated = 1
        # logger.info("setting epsilon for truncation equal to 1 - starting fine-tuning with all space policy")

        self.update_temperature(i_episode)
        if i_episode == self.curriculum:
            print(self.env.answer_sampling)
            logger.info("UPDATING ANSWER SAMPLING FROM RANDOM TO UNIFORM...")
            self.env.update_mode(mode=self.env.mode, answer_sampl="uniform")
            print(self.env.answer_sampling)

    def update_temperature(self, i_episode):
        if i_episode + 1 == self.inv_schedule_step:
            self.temp_factor = 1 / self.temp_factor
            print("inversing the temperature schedule at episode {}".format(i_episode + 1))
        if (i_episode + 1) >= self.schedule_start:
            if (i_episode + 1) == self.schedule_start:
                print("starting the temperature scheduling at episode {}".format(i_episode + 1))
            if self.temp_factor < 1:
                if (i_episode + 1) % self.temperature_step == 0 and self.temperature > self.temperature_min:
                    self.temperature *= self.temp_factor
                    if self.temperature < self.temperature_min:
                        logger.info("LAST TEMPERATURE UPDATE at temp {}".format(self.temperature_min))
                        self.temperature = self.temperature_min
            else:
                if (i_episode + 1) % self.temperature_step == 0 and self.temperature < self.temperature_max:
                    self.temperature *= self.temp_factor
                    if self.temperature > self.temperature_max:
                        logger.info("LAST TEMPERATURE UPDATE at temp {}".format(self.temperature_max))
                        self.temperature = self.temperature_max
        self.writer.add_scalar('temperature', self.temperature, i_episode)

    def act(self, state, mode='sampling', truncation=True, forced=None, ht=None, ct=None):
        valid_actions, action_probs, logits_lm, log_probas_lm, origin_log_probs_lm = self.truncation.get_valid_actions(
            state, truncation, temperature=self.temperature)
        alpha = self.alpha_logits_lm
        policy_dist, policy_dist_truncated, value, ht, ct = self.get_policy_distributions(state, valid_actions,
                                                                                          logits_lm,
                                                                                          alpha=alpha, ht=ht, ct=ct)
        if self.truncation_optim == 1:
            policy_dist = policy_dist_truncated
        action = self.sample_action(policy_dist=policy_dist, policy_dist_truncated=policy_dist_truncated,
                                    valid_actions=valid_actions, mode=mode, forced=forced)
        log_prob = policy_dist.log_prob(action.to(self.device)).view(-1)
        log_prob_truncated = policy_dist_truncated.log_prob(action.to(self.device)).view(-1)

        return action, log_prob, value, (
            valid_actions, action_probs,
            log_prob_truncated), policy_dist, logits_lm, log_probas_lm, origin_log_probs_lm, ht, ct

    def get_policy_distributions(self, state, valid_actions, logits_lm=None, alpha=0., ht=None, ct=None):
        policy_dist, policy_dist_truncated, value, ht, ct = self.policy(state.text, state.img, state.answer,
                                                                        valid_actions=valid_actions,
                                                                        logits_lm=logits_lm, alpha=alpha, ht=ht, ct=ct)
        return policy_dist, policy_dist_truncated, value, ht, ct

    def sample_action(self, policy_dist, policy_dist_truncated, valid_actions, mode='sampling', forced=None):
        policy_to_sample_from = policy_dist_truncated
        epsilon_truncated_sample = random.random()
        if epsilon_truncated_sample < self.epsilon_truncated:
            policy_to_sample_from = policy_dist
        if mode == 'forced':
            action = forced
        elif mode == 'sampling':
            action = policy_to_sample_from.sample()
        elif mode == 'greedy':
            action = torch.argmax(policy_to_sample_from.probs).view(1).detach()
        if policy_to_sample_from.probs.size() != policy_dist.probs.size():
            action = torch.gather(valid_actions, 1, action.view(1, 1))
        return action

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

    def test(self, num_episodes=10, test_mode='sampling', test_seed=0, num_diversity=1):
        for env in self.test_envs:
            logger.info('-----------------------Starting Evaluation for {} dialog ------------------'.format(env.mode))
            self.test_env(env, num_episodes=num_episodes, test_mode=test_mode, test_seed=test_seed)

    def init_hidden(self, state):
        h, c = self.policy.init_hidden_state(state)
        return h, c

    def generate_one_episode(self, timestep, i_episode, env, seed=None, train=True, truncation=True,
                             test_mode='sampling', metrics=[], idx_diversity=0, num_diversity=10):
        if train or seed is None:
            state, ep_reward = env.reset(seed=seed), 0
        else:
            state, ep_reward = env.reset(i_episode=i_episode), 0
        (ht, ct) = self.init_hidden(state)
        for t in range(0, env.max_len):
            forced = env.ref_question[t]
            action, log_probs, value, (
                valid_actions, actions_probs,
                log_probs_truncated), dist, logits_lm, log_probas_lm, origin_log_probs_lm, new_ht, new_ct = self.act(
                state=state,
                mode=test_mode,
                truncation=truncation,
                forced=forced, ht=ht, ct=ct)
            new_state, (reward, closest_question, pred_answer), done, _ = env.step(action.cpu().numpy())
            if train:
                # Saving reward and is_terminal:
                self.memory.add_step(action, state.text[0], state.img[0], log_probs, log_probs_truncated, reward, done,
                                     value, state.answer, ht, ct, log_probas_lm)
                if self.env.reward_type == "vilbert" and done:
                    self.writer.add_scalar("vilbert_rank", pred_answer, i_episode)
            timestep += 1
            for key, metric in metrics.items():
                metric.fill(state=state, action=action, done=done, dist=dist, valid_actions=valid_actions,
                            actions_probs=actions_probs, ref_question=env.ref_questions,
                            ref_questions_decoded=env.ref_questions_decoded, reward=reward,
                            closest_question=closest_question, new_state=new_state, log_probs=log_probs,
                            log_probs_truncated=log_probs_truncated, test_mode=test_mode, pred_answer=pred_answer,
                            i_episode=i_episode, ref_question_idx=env.ref_question_idx, logits_lm=logits_lm,
                            log_probas_lm=log_probas_lm, timestep=t, origin_log_probs_lm=origin_log_probs_lm,
                            alpha=self.alpha_logits_lm, ref_answer=env.ref_answer)
            state = new_state
            ht = new_ht
            ct = new_ct
            ep_reward += reward

            # update if its time
            if train:
                if self.update_mode == "step" and timestep % self.update_every == 0:
                    loss = self.update()
                    logger.info("UPDATING POLICY WEIGHTS...")
                    self.memory.clear_memory()
                    timestep = 0
                else:
                    loss = None

            if done:
                if train:
                    if self.update_mode == "episode" and i_episode % self.update_every == 0:
                        loss = self.update()
                        logger.info("UPDATING POLICY WEIGHTS...")
                        self.memory.clear_memory()
                else:
                    loss = None
                break
        for key, metric in metrics.items():
            metric.compute(state=state, closest_question=closest_question, img_idx=env.img_idx, reward=reward,
                           ref_question=env.ref_questions, ref_questions_decoded=env.ref_questions_decoded,
                           question_idx=env.ref_question_idx, test_mode=test_mode, pred_answer=pred_answer,
                           ref_answer=env.ref_answer, idx_diversity=idx_diversity, num_diversity=num_diversity)

        return state, ep_reward, closest_question, valid_actions, timestep, loss

    def test_env(self, env, num_episodes=10, test_mode='sampling', test_seed=0):
        num_diversity = 10 if test_mode == "sampling_ranking_lm" else 1
        test_mode_episode = {"greedy": "greedy", "sampling": "sampling", "sampling_ranking_lm": "sampling"}
        print("temperature at test: {}".format(self.temperature))
        env.reset()  # init env.
        timestep = 1
        self.policy.eval()
        for i_episode in range(num_episodes):
            logger.info('-' * 20 + 'Test Episode: {}'.format(i_episode) + '-' * 20)
            seed = i_episode if test_seed else None
            for key_trunc, trunc in self.eval_trunc.items():
                metrics = self.get_metrics(env.mode, key_trunc, test_mode)
                for i in range(num_diversity):  # loop multiple time over the same image to measure langage diversity.
                    with torch.no_grad():
                        state, ep_reward, closest_question, valid_actions, timestep, _ = self.generate_one_episode(
                            timestep=timestep, i_episode=i_episode, env=env, seed=seed, train=False,
                            test_mode=test_mode_episode[test_mode],
                            truncation=trunc, metrics=metrics, idx_diversity=i, num_diversity=num_diversity)
                    for _, metric in metrics.items():
                        metric.write()
                        metric.log(valid_actions=valid_actions)
                for _, metric in metrics.items():
                    metric.write_div()
        for key_trunc in self.eval_trunc.keys():
            metrics = self.get_metrics(env.mode, key_trunc, test_mode)
            idx_to_keep = None
            if test_mode == "sampling_ranking_lm":
                language_score = metrics["language_score"]
                idx_to_keep = language_score.get_min_ppl_idxs(num_diversity)
            for key_metric, metric in metrics.items():
                metric.post_treatment(num_episodes=num_episodes, idx_to_keep=idx_to_keep)

    def log_at_train(self, i_episode, ep_reward, state, closest_question, valid_actions):
        logger.info('-' * 20 + 'Episode {} - Img  {}'.format(i_episode, self.env.img_idx) + '-' * 20)
        logger.info('Last reward: {:.2f}'.format(ep_reward))
        for key, metric in self.metrics["train"].items():
            metric.log(valid_actions=valid_actions)
            metric.write()
        logger.info("-" * 100)

    def learn(self, num_episodes=100):
        sampling_mode = "forced" if self.pretrain else "sampling"
        start_time = time.time()
        current_time = time.time()
        timestep = 1
        for i_episode in range(self.start_episode, self.start_episode + num_episodes):
            seed = i_episode if self.train_seed else None
            state, ep_reward, closest_question, valid_actions, timestep, loss = self.generate_one_episode(
                timestep=timestep, i_episode=i_episode, env=self.env, seed=seed,
                metrics=self.metrics["train"], test_mode=sampling_mode)
            self.update_per_episode(i_episode=i_episode, num_episodes_train=num_episodes)
            if i_episode % self.log_interval == 0:
                self.log_at_train(i_episode=i_episode, ep_reward=ep_reward, state=state,
                                  closest_question=closest_question, valid_actions=valid_actions)

            if i_episode % 1000 == 0:
                elapsed = time.time() - current_time
                logger.info("Training time for 1000 episodes: {:5.2f}".format(elapsed))
                current_time = time.time()
                # saving checkpoint:
                self.save_ckpt(EPOCH=i_episode, loss=loss)

        if valid_actions is not None and "action_probs" in self.metrics["train"] and "action_probs_lm" in self.metrics[
            "train"]:  # to compare the discrepancy between the 'truncated policy' and the 'all space' policy
            self.writer.add_custom_scalars({'Train_all_probs': {'action_probs': ['Multiline', ['train_action_probs',
                                                                                               'train_action_probs_truncated',
                                                                                               'train_action_probs_lm']]}})

        for _, metric in self.metrics["train"].items():
            metric.post_treatment(num_episodes=num_episodes)
        logger.info("total training time: {:7.2f}".format(time.time() - start_time))
        logger.info(
            "--------------------------------------------END OF TRAINING ----------------------------------------------------")

    def compute_write_all_metrics(self, output_path, logger):
        # write to csv test scalar metrics:
        logger.info(
            "------------------------------------- test metrics statistics -----------------------------------------")
        all_metrics = {trunc: {} for trunc in self.eval_trunc.keys()}
        for key in self.test_metrics_names:
            stats_dict = {trunc: {} for trunc in self.eval_trunc.keys()}
            stats_dict_div = {trunc: {} for trunc in self.eval_trunc.keys()}

            instances_of_metric = [self.metrics[key_mode][key] for key_mode in self.metrics.keys() if
                                   key_mode != "train"]
            # for stats
            for metric in instances_of_metric:
                if metric.stats:
                    for key_stat, stat in metric.stats.items():
                        stats_dict[metric.trunc]["_".join([metric.env_mode, metric.sampling, key_stat])] = stat[0]
                        if str(stat[0]) != 'nan':
                            all_metrics[metric.trunc].setdefault(key_stat, []).append(stat[0])

                if metric.stats_div:
                    for key_stat, stat in metric.stats.items():
                        stats_dict[metric.trunc]["_".join([metric.env_mode, metric.sampling, key_stat])] = stat[0]
                        # all_metrics[metric.trunc].setdefault(key_stat, []).append(stat[0])
            stats_path = os.path.join(self.out_path, "stats", "{}.csv".format(key))
            div_path = os.path.join(self.out_path, "stats", "{}_div.csv".format(key))

            pd.DataFrame(data=stats_dict).to_csv(stats_path)
            pd.DataFrame(data=stats_dict_div).to_csv(div_path)

        # for all metrics
        for trunc in all_metrics.keys():
            for key_s in all_metrics[trunc].keys():
                if len(all_metrics[trunc][key_s]) > 0:
                    all_metrics[trunc][key_s] = np.round(np.mean(all_metrics[trunc][key_s]), decimals=3)

        stats_path = os.path.join(self.out_path, "all_metrics.csv")
        pd.DataFrame(data=all_metrics).to_csv(stats_path)
