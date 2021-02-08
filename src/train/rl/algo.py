# https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
import copy
import datetime
import json
import logging
import os
import time

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from RL_toolbox.reward import Bleu_sf0
from RL_toolbox.reward import Bleu_sf2
from train.rl.truncation import mask_inf_truncature, truncations
# from train.train_functions import evaluate_policy
from utils.utils_train import create_logger, write_to_csv

logger = logging.getLogger()


class RLAlgo:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args, lm, max_len, alpha_lm, truncate_mode,
                 truncation_params=None, is_correction=False, baseline=True, s_min=1, s_max=-1):
        self.lm = lm
        self.model = model
        self.dataset_name = args.dataset
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        self.train_generator = DataLoader(dataset=train_dataset, batch_size=args.bs, drop_last=True,
                                          num_workers=args.num_workers)
        self.val_generator = DataLoader(dataset=val_dataset, batch_size=args.bs, drop_last=True,
                                        num_workers=args.num_workers)
        self.batch_size = args.bs
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr)
        PAD_IDX = train_dataset.vocab_questions["<PAD>"]
        self.criterion = torch.nn.NLLLoss(ignore_index=PAD_IDX)
        self.mse = torch.nn.MSELoss(reduction="none")

        self.EPOCHS = args.ep
        self.grad_clip = args.grad_clip if args.grad_clip is not None else 5.
        self.print_interval = args.print_interval
        self.device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")
        self.create_out_path(args)
        self.writer = SummaryWriter(log_dir=os.path.join(self.out_path, "runs"))

        self.train_function, self.eval_function = self.get_algo_functions(args)
        self.task = args.task
        self.check_batch()
        self.s_min = s_min
        self.s_max = s_max
        self.truncation_params = truncation_params
        self.truncation = truncations[truncate_mode](pretrained_lm=self.lm,
                                                     s_min=s_min, s_max=s_max,
                                                     truncation_params=truncation_params)  # adding the truncation class.
        self.reward_function = Bleu_sf2()
        self.gamma = 0.99
        self.max_len = max_len
        self.alpha_lm = alpha_lm
        self.language_metrics = {k: v for k, v in zip(["bleu-1", "bleu-2", "bleu"],
                                                      [Bleu_sf0(sf_id=args.bleu_sf, n_gram=2),
                                                       Bleu_sf0(sf_id=args.bleu_sf, n_gram=3),
                                                       Bleu_sf0(sf_id=args.bleu_sf, n_gram=4)])}
        self.writer_iteration = 0
        self.is_correction = is_correction
        self.baseline = baseline
        self.rl_all, self.vf_all, self.rewards_all, self.ranks_all = [], [], [], []
        self.dialog_all, self.in_va_all, self.log_probs_truncated_all = [], [], []
        self.episode_idx = 0
        self.update_iteration = 1

    def create_out_path(self, args):
        if args.model_path is not None:
            out_path = os.path.join(args.model_path, "eval_from_loaded_model")
            self.out_path = os.path.join(out_path,
                                         "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        else:
            out_path = '{}_{}_{}_layers_{}_emb_{}_hidden_{}_pdrop_{}_gradclip_{}_bs_{}_lr_{}'.format(args.dataset,
                                                                                                     args.task,
                                                                                                     args.model,
                                                                                                     args.num_layers,
                                                                                                     args.emb_size,
                                                                                                     args.hidden_size,
                                                                                                     args.p_drop,
                                                                                                     args.grad_clip,
                                                                                                     args.bs, args.lr)
            if args.task == 'policy':
                out_path = out_path + '_cond-answer_{}'.format(args.condition_answer)
            self.out_path = os.path.join(args.out_path, out_path,
                                         "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        if not os.path.exists(self.out_path):
            os.makedirs(self.out_path)
        out_file_log = os.path.join(self.out_path, 'training_log.log')
        self.logger = create_logger(out_file_log)
        self.out_csv = os.path.join(self.out_path, 'train_history.csv')
        self.out_lm_metrics = os.path.join(self.out_path, 'lm_metrics_sf{}.csv'.format(args.bleu_sf))
        self.model_path = os.path.join(self.out_path, 'model.pt')
        self.logger.info("hparams: {}".format(vars(args)))
        self.logger.info('train dataset length: {}'.format(self.train_dataset.__len__()))
        self.logger.info("val dataset length: {}".format(len(self.val_dataset)))
        if self.dataset_name == "vqa":
            self.logger.info("number of filtered entries:{}".format(len(self.train_dataset.filtered_entries)))
        self.logger.info('number of tokens: {}'.format(self.train_dataset.len_vocab))
        self._save_hparams(args, self.out_path)

    def _save_hparams(self, args, out_path):
        dict_hparams = vars(args)
        dict_hparams = {key: str(value) for key, value in dict_hparams.items()}
        config_path = os.path.join(out_path, "config.json")
        with open(config_path, 'w') as fp:
            json.dump(dict_hparams, fp, sort_keys=True, indent=4)

    def check_batch(self):
        if self.task == 'policy':
            (temp_inp, temp_tar), answers, img = self.train_dataset.__getitem__(0)
            if isinstance(img, tuple):
                feats = img[0]
            else:
                feats = img
            self.logger.info('img features shape:{}'.format(feats.shape))
        elif self.task == 'lm':
            if self.dataset_name == "clevr":
                temp_inp, temp_tar = self.train_dataset.__getitem__(0)
            elif self.dataset_name == "vqa":
                (temp_inp, temp_tar), _, _ = self.train_dataset.__getitem__(0)

        self.logger.info("input shape: {}".format(temp_inp.shape))
        self.logger.info("target shape: {}".format(temp_tar.shape))
        self.logger.info("input:{}".format(temp_inp))
        self.logger.info("target: {}".format(temp_tar))

    def get_algo_functions(self, args):
        train_function = self.train_one_epoch_policy
        eval_function = self.evaluate_policy
        return train_function, eval_function

    def get_answers_img_features(self, dataset, index):
        entry = dataset.filtered_entries[index]
        img_feats, _, _ = dataset.get_img_data(entry)
        answer, _ = dataset.get_answer_data(entry)
        return img_feats, answer

    def train(self):
        self.logger.info("start training...")
        train_loss_history, train_ppl_history, val_loss_history, val_ppl_history = [], [], [], []
        best_val_loss = None
        for epoch in range(self.EPOCHS):
            self.logger.info('epoch {}/{}'.format(epoch + 1, self.EPOCHS))
            train_loss, elapsed = self.train_function(train_generator=self.train_generator,
                                                      optimizer=self.optimizer,
                                                      criterion=self.criterion,
                                                      device=self.device,
                                                      grad_clip=self.grad_clip,
                                                      print_interval=self.print_interval)
            self.logger.info('train loss {:5.3f} '.format(train_loss))
            self.logger.info('time for one epoch...{:5.2f}'.format(elapsed))
            # val_loss = self.eval_function(model=self.model, val_generator=self.val_generator, criterion=self.criterion,
            #                              device=self.device)
            # self.logger.info('val loss: {:5.3f} - val perplexity: {:8.3f}'.format(val_loss, math.exp(val_loss)))

            # saving loss and metrics information.
            train_loss_history.append(train_loss)
            # train_ppl_history.append(math.exp(train_loss))
            # val_loss_history.append(val_loss)
            # val_ppl_history.append(math.exp(val_loss))
            self.logger.info('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or train_loss < best_val_loss:
                with open(self.model_path, 'wb') as f:
                    torch.save(self.model, f)
                best_val_loss = train_loss

        self.logger.info("saving loss and metrics information...")
        self.save_results()
        hist_keys = ['train_loss', 'train_ppl', 'val_loss', 'val_ppl']
        hist_dict = dict(zip(hist_keys, [train_loss_history, train_ppl_history, val_loss_history, val_ppl_history]))
        write_to_csv(self.out_csv, hist_dict)

    def _sample_(self, input, num_words=20, img_feats=None,
                 index_img=None,
                 answer=None, write=True):
        answer_ = answer[0].cpu().item() if answer is not None else answer
        input_idx = input
        with torch.no_grad():
            for i in range(num_words):
                img_feats = img_feats.to(self.device)
                answer = answer.to(self.device)
                logits, _ = self.model(state_text=input_idx, state_img=img_feats,
                                       state_answer=answer)  # output = logits (S, num_tokens)
                word_idx = logits[-1].squeeze().argmax()
                input_idx = torch.cat([input_idx, word_idx.view(1, 1)], dim=-1)
            words = self.val_dataset.question_tokenizer.decode(input_idx.squeeze().cpu().numpy())
        dict_words = words
        if write:
            out_file_generate = os.path.join(self.out_path,
                                             'generate_words_img_{}_answer_{}.txt'.format(index_img, answer_))
            with open(out_file_generate, 'w') as f:
                f.write(dict_words)
                f.close()

        return dict_words

    def _generate_text(self, input, temperatures=["greedy", 0.5, 1, 2], num_words=20, img_feats=None, index_img=None,
                       answer=None, write=True):
        dict_words = {k: [] for k in temperatures}
        for temp in temperatures:
            answer_ = answer[0].cpu().item() if answer is not None else answer
            input_idx = input
            with torch.no_grad():
                for i in range(num_words):
                    img_feats = img_feats.to(self.device)
                    answer = answer.to(self.device)
                    logits, _ = self.model(state_text=input_idx, state_img=img_feats,
                                           state_answer=answer)  # output = logits (S, num_tokens)

                    word_idx = logits[:, -1].squeeze().argmax()
                    input_idx = torch.cat([input_idx, word_idx.view(1, 1)], dim=-1)
                words = self.val_dataset.question_tokenizer.decode(input_idx.squeeze().cpu().numpy())
            dict_words[temp] = words
            if write:
                out_file_generate = os.path.join(self.out_path,
                                                 'generate_words_temp_{}_img_{}_answer_{}.txt'.format(temp, index_img,
                                                                                                      answer_))
                with open(out_file_generate, 'w') as f:
                    f.write(dict_words[temp])
                    f.close()

        return dict_words

    def generate_text(self, temperatures=["greedy", 0.5, 1, 2], words=20):
        input = self.test_dataset.vocab_questions["<SOS>"]
        input = torch.LongTensor([input]).view(1, 1).to(self.device)
        indexes = list(range(3))
        for index in indexes:
            print("Generating text conditioned on img: {}".format(index))
            img_feats, answer = self.get_answers_img_features(dataset=self.val_dataset, index=index)
            img_feats = img_feats.unsqueeze(0)
            _ = self._generate_text(input, temperatures=temperatures, num_words=words, img_feats=img_feats,
                                    index_img=index,
                                    answer=answer)

    def evaluate_policy(self, val_generator, criterion, device):
        self.model.eval()  # turn on evaluation mode which disables dropout.
        total_loss = 0.
        start_time = time.time()
        with torch.no_grad():
            for batch, ((inputs, targets), answers, img) in enumerate(val_generator):
                if isinstance(img, list):
                    feats = img[0]
                else:
                    feats = img
                answers = answers.squeeze()
                inputs, feats, answers = inputs.to(device), feats.to(device), answers.to(device)
                targets = targets.view(targets.size(1) * targets.size(0)).to(device)
                logits, _ = self.model(inputs, feats, answers)
                logits = logits.view(-1, logits.size(-1))
                log_probs = F.log_softmax(logits, dim=-1)
                total_loss += criterion(log_probs, targets).item()
        print("Evaluation time {:5.2f}".format(time.time() - start_time))
        return total_loss / (batch + 1)

    def compute_language_metrics(self, temperatures):
        """
        Compute different versions of BLEU: try smoothing techniques seven at first. then 5, 6.
        METEOR
        :return:
        """
        input = self.test_dataset.vocab_questions["<SOS>"]
        input = torch.LongTensor([input]).view(1, 1).to(self.device)
        dict_metrics = {k: 0. for k in self.language_metrics.keys()}
        result_metrics = {k: dict_metrics for k in temperatures}
        for ((inputs, targets), answers, img) in self.val_generator:
            if isinstance(img, list):
                feats = img[0]
            else:
                feats = img
            for i in range(targets.shape[0]):
                question_decoded = [self.val_dataset.question_tokenizer.decode(targets[i].cpu().numpy())]
                num_words = len(question_decoded[0].split(" ")) + 1
                if self.task == "lm":
                    dict_questions = self._generate_text(input, temperatures=temperatures, num_words=num_words,
                                                         write=False)
                elif self.task == "policy":
                    dict_questions = self._generate_text(input, temperatures=temperatures,
                                                         img_feats=feats[i].unsqueeze(0), answer=answers[i].view(1),
                                                         num_words=num_words, write=False)
                for temp in temperatures:
                    for name, metric in self.language_metrics.items():
                        result, _, _ = metric.get(dict_questions[temp], question_decoded, step_idx=None,
                                                  done=True)
                        result_metrics[temp][name] = result_metrics[temp][name] + result
        # getting the average
        result_metrics = {k: {k_: v_ / len(self.val_dataset) for k_, v_ in v.items()} for k, v in
                          result_metrics.items()}
        df = pd.DataFrame.from_dict(result_metrics)
        df.to_csv(self.out_lm_metrics, index_label="metrics", columns=temperatures)
        return result_metrics

    def update(self, log_probs, log_probs_truncated, gts, inputs, targets, feats, answers):

        self.model.zero_grad()
        logits, values = self.model(state_text=inputs, state_img=feats,
                                    state_answer=answers)
        values = values.squeeze()
        log_probs_all = F.log_softmax(logits, dim=-1)
        log_probs_actions = log_probs_all.gather(-1, targets.unsqueeze(dim=-1)).view(
            log_probs_all.size(0), log_probs_all.size(1))

        advs = gts.clone()
        if self.baseline:
            advs -= values.cpu().detach().view(gts.size(0), gts.size(1))

        log_probs_advs = log_probs_actions * advs
        if self.is_correction:
            is_ratios = torch.exp(log_probs.detach() - log_probs_truncated.detach())
            cumprod_ratios = torch.cumprod(is_ratios, dim=-1)
            log_probs_advs *= cumprod_ratios
        rl_loss_per_episode = -log_probs_advs.sum(dim=1)
        rl_loss = rl_loss_per_episode.mean()
        value_loss = torch.square(gts.view(-1) - values.view(-1)).sum()

        loss = rl_loss
        if self.baseline:
            loss += 0.5 * value_loss

        self.optimizer.zero_grad()
        loss.mean().backward()

        clip_grad_norm_(self.model.parameters(), self.grad_clip)
        self.optimizer.step()
        return loss, rl_loss, value_loss

    def train_one_epoch_policy(self, train_generator, optimizer, criterion, device, grad_clip,
                               print_interval=10):
        self.model.train()  # Turns on train mode which enables dropout.
        total_loss = 0.
        start_time = time.time()
        start_time_epoch = time.time()

        for batch, ((inputs, targets), answers, img) in enumerate(train_generator):
            if isinstance(img, list):
                feats = img[0]
            else:
                feats = img
            answers = answers.squeeze()
            inputs, feats, answers = inputs.to(device), feats.to(device), answers.to(device)
            inputs_ = inputs[:, 0:1].to(device)
            log_probs = torch.zeros((inputs.size(0), self.max_len)).to(self.device)
            ranks = torch.zeros_like(log_probs)
            in_valid_actions = torch.zeros_like(log_probs)
            log_probs_truncated = torch.zeros_like(log_probs)

            with torch.no_grad():
                for t in range(self.max_len):
                    log_probas_lm, logits_lm, origin_log_probs_lm = self.lm.forward(inputs_, 1.)
                    mask_valid_actions = self.truncation.get_mask(log_probas_lm, logits_lm)

                    sort_lm, sort_lm_ind = torch.sort(logits_lm, descending=True)
                    ranks[:, t] = (sort_lm_ind == targets[:, t].view(-1, 1)).nonzero()[:, -1]
                    # in_valid_actions[:, t] = (targets[:, t].view(-1, 1) == valid_actions).sum(dim=1)
                    logits_, _ = self.model(state_text=inputs_, state_img=feats,
                                            state_answer=answers)  # output = logits (S, num_tokens)
                    last_logits = (1 - self.alpha_lm) * logits_[:, -1, :] + self.alpha_lm * logits_lm
                    probs = F.softmax(last_logits, dim=-1)
                    dist = Categorical(probs)

                    dist_truncated = mask_inf_truncature(mask_valid_actions, logits_[:, -1, :], self.device,
                                                         logits_.size(-1))
                    sort_probs, sort_ind = torch.sort(dist_truncated.probs, descending=True)
                    sort_words = [self.train_dataset.question_tokenizer.decode(sort_ind[j, :10].numpy()).split() for j
                                  in range(sort_ind.size(0))]
                    logger.info("sort words {}".format(sort_words))
                    logger.info("sort probs {}".format(sort_probs[:, :10]))
                    actions = dist_truncated.sample().to(self.device)
                    log_probs_truncated[:, t] = torch.log(dist_truncated.probs.gather(-1, actions.view(-1, 1)).view(-1))
                    prob_actions = dist.probs.to(self.device).gather(-1, actions.view(-1, 1)).view(-1)
                    log_probs[:, t] = torch.log(prob_actions).to(self.device)
                    inputs_ = torch.cat([inputs_.to(device), actions.view(-1, 1)], dim=-1)

            gts, rewards, dialog, targets_dialog = self.compute_rewards(inputs_, targets)

            # estimate the loss using one MonteCarlo rollout
            inputs_sampled, targets_sampled = inputs_[:, :-1], inputs_[:, 1:]
            loss, rl_loss, value_loss = self.update(log_probs, log_probs_truncated, gts, inputs_sampled,
                                                    targets_sampled, feats, answers)

            total_loss += loss.mean().item()

            self.dialog_all.append([dialog[0], targets_dialog[0]])
            self.rewards_all.append(np.mean(rewards))
            self.vf_all.append(value_loss.detach().item())
            self.rl_all.append(rl_loss.detach().item())
            self.log_probs_truncated_all.append(log_probs_truncated.detach().tolist())
            self.in_va_all.append(in_valid_actions.mean(dim=0).tolist())
            ranks_median, _ = ranks.median(dim=-1)
            self.ranks_all.append(ranks_median.view(1, -1))

            # print loss every number of batches
            if (batch + 1) % print_interval == 0:
                print('loss for batch {}: {:5.3f}'.format(batch + 1, total_loss / (batch + 1)))
                print('time for {} training steps: {:5.2f}'.format(print_interval, time.time() - start_time))
                logger.debug('rl loss {}'.format(np.mean(self.rl_all[-print_interval:])))
                logger.debug('value loss {}'.format(np.mean(self.vf_all[-print_interval:])))
                logger.info("rewards:{}".format(np.mean(self.rewards_all[-print_interval:])))
                logger.info("convergence :{}".format(np.exp(np.mean(self.log_probs_truncated_all[-print_interval:]))))
                first_log_probs_tr = np.array(self.log_probs_truncated_all[-print_interval:])[:, :, 0]
                logger.info(" 1rst prob trunc :{}".format(np.exp(np.mean(first_log_probs_tr))))

                logger.info("dialog:{}".format(dialog[0]))
                logger.info("true dialog:{}".format(targets_dialog[0]))
                ranks_medians = torch.cat(self.ranks_all[-print_interval:], dim=0)
                logger.debug("ranks :{}".format(ranks_medians))
                self.writer.add_scalar("rewards", np.mean(rewards), self.writer_iteration)
                self.writer.add_scalar("vf_loss", value_loss.detach().item(), self.writer_iteration)
                self.writer.add_scalar("rl_loss", rl_loss.detach().item(), self.writer_iteration)
                self.writer_iteration += 1

                start_time = time.time()

            curr_loss = total_loss / (batch + 1)
            elapsed = time.time() - start_time_epoch
            self.episode_idx += 1

        return curr_loss, elapsed

    def compute_rewards(self, inputs_, targets):
        dialog = [self.train_dataset.question_tokenizer.decode(question) for question in
                  inputs_.cpu().numpy()]
        targets_dialog = [self.train_dataset.question_tokenizer.decode(question[:self.max_len]) for question
                          in targets.cpu().numpy()]

        rewards = [self.reward_function.get(dialog[t_], [targets_dialog[t_]], done=True)[0] for t_ in
                   range(len(dialog))]

        rewards_ = torch.zeros((inputs_.size(0), inputs_.size(1) - 1))
        rewards_[:, -1] = torch.tensor(rewards).view(-1)
        gts = torch.zeros((inputs_.size(0), inputs_.size(1) - 1))

        discounted_reward = 0
        for timestep in range(self.max_len):
            discounted_reward = rewards_[:, -timestep - 1] + (self.gamma * discounted_reward)
            gts[:, -timestep - 1] = discounted_reward

        return gts, rewards, dialog, targets_dialog

    def save_results(self):
        pd.DataFrame(self.rewards_all).to_csv(os.path.join(self.out_path, "rewards.csv"), index=False, header=False)
        pd.DataFrame(self.rl_all).to_csv(os.path.join(self.out_path, "rl_losses.csv"), index=False, header=False)
        pd.DataFrame(self.vf_all).to_csv(os.path.join(self.out_path, "vf_losses.csv"), index=False, header=False)
        pd.DataFrame(self.dialog_all).to_csv(os.path.join(self.out_path, "dialogs.csv"), index=False, header=False)
        pd.DataFrame(self.in_va_all).to_csv(os.path.join(self.out_path, "in_vas.csv"), index=False, header=False)


class PPO_algo(RLAlgo):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.K_epochs = 1
        self.eps_clip = 0.02
        self.new_model = copy.deepcopy(self.model)
        self.new_model.load_state_dict(self.model.state_dict())
        self.optimizer = torch.optim.Adam(params=self.new_model.parameters(), lr=self.lr)
        self.entropy_coeff = 0.01
        self.update_iteration = 1

    def update(self, log_probs, log_probs_truncated, gts, inputs, targets, feats, answers):

        self.model.zero_grad()
        old_logits, old_values = self.model(state_text=inputs, state_img=feats,
                                            state_answer=answers)
        old_values = old_values.squeeze()
        old_log_probs_all = F.log_softmax(old_logits, dim=-1)
        old_log_probs_actions = old_log_probs_all.gather(-1, targets.unsqueeze(dim=-1)).view(
            old_log_probs_all.size(0), old_log_probs_all.size(1))

        advs = gts.clone()
        if self.baseline:
            advs -= old_values.cpu().detach().view(gts.size(0), gts.size(1))

        for _ in range(self.K_epochs):
            self.model.zero_grad()
            logits, values = self.new_model(state_text=inputs, state_img=feats,
                                            state_answer=answers)
            values = values.squeeze()
            log_probs_all = F.log_softmax(logits, dim=-1)
            log_probs_actions = log_probs_all.gather(-1, targets.unsqueeze(dim=-1)).view(
                log_probs_all.size(0), log_probs_all.size(1))
            ratios = torch.exp(log_probs_actions - old_log_probs_actions.detach())
            if self.is_correction:
                # computing the Importance Sampling ratio (pi_theta_old / rho_theta_old)
                is_ratios = torch.exp(log_probs - log_probs_truncated.to(self.device))
                ratios = ratios * is_ratios
            surr1 = ratios * advs
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advs
            surr = -torch.min(surr1, surr2)
            entropy_loss = self.entropy_coeff * Categorical(torch.exp(log_probs_all)).entropy()
            vf_loss = 0.5 * self.mse(values.squeeze(), gts)
            loss = surr.sum(dim=1).mean() + vf_loss.sum() - entropy_loss.sum()
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            clip_grad_norm_(self.new_model.parameters(), self.grad_clip)
            self.optimizer.step()
        # return loss, rl_loss, value_loss
        self.model.load_state_dict(self.new_model.state_dict())
        return loss, surr.mean(), vf_loss.mean()
