import logging
import os

import h5py
import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import sentence_bleu
from torch.nn.utils.rnn import pad_sequence

from utils.utils_train import write_to_csv, write_to_csv_by_row


class Metric:
    def __init__(self, agent, train_test):
        self.measure = []
        self.metric = []
        self.metric_history = []
        self.idx_step = 0
        self.idx_word = 0
        self.idx_write = 1
        self.agent = agent
        self.train_test = train_test
        self.dict_metric, self.dict_stats = {}, {}  # for csv writing.

    def fill(self, **kwargs):
        self.fill_(**kwargs)
        self.idx_word += 1
        self.idx_step += 1

    def compute(self, **kwargs):
        self.compute_(**kwargs)
        self.measure = []
        self.idx_word = 0
        self.idx_step = 0

    def reset(self):
        self.idx_word = 0

    def reinit_train_test(self, train_test):
        self.train_test = train_test

    def write(self, **kwargs):
        if self.type == "scalar":
            self.agent.writer.add_scalar(self.train_test + "_" + self.key, np.mean(self.metric), self.idx_write)
        else:
            self.agent.writer.add_text(self.train_test + "_" + self.key, '  \n'.join(self.metric[-1:]), self.idx_write)
        self.idx_write += 1
        self.metric_history.extend(self.metric)
        self.metric = []

    def log(self, **kwargs):
        pass

    def write_to_csv(self):
        if self.dict_metric:
            for key, value in self.dict_metric.items():
                self.dict_stats[key] = [np.round(np.mean(value), decimals=3), np.round(np.std(value), decimals=3),
                                        np.round(len(value))]
                logging.info('{}: {} +/- {}'.format(key, np.round(np.mean(value), decimals=3),
                                                    np.round(np.std(value), decimals=3)))
            # write_to_csv(self.out_csv_file + '.csv', self.dict_metric)
            write_to_csv(self.out_csv_file + '_stats.csv', self.dict_stats)

    def post_treatment(self):
        pass


# ----------------------------------  TRAIN METRICS -------------------------------------------------------------------------------------

class VAMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "text"
        self.key = "valid_actions"

    def fill_(self, **kwargs):
        state_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["state"].text.numpy()[0], ignored=['<PAD>'])
        if kwargs["valid_actions"] is not None:
            top_words_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["valid_actions"].cpu().numpy()[0])
            weights_words = ["{}/{:.3f}".format(word, weight, number=3) for word, weight in
                             zip(top_words_decoded.split(), kwargs["actions_probs"].cpu().detach().numpy()[0])]
            string = "next possible words for {} : {}".format(state_decoded, ", ".join(weights_words))
        else:
            string = ""
        self.measure.append(string)

    def compute_(self, **kwargs):
        self.metric = self.measure
        pass

    def log(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            logging.info('---------------------Valid action space------------------------------')
            logging.info('\n'.join(self.metric))
            logging.info('---------------------------------------------------------------------')
        else:
            pass

    def write(self):
        pass


class SizeVAMetric(Metric):
    '''Compute the average size of the truncated action space during training for truncation functions proba_thr & sample_va'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.counter = 0
        self.key = "size_valid_actions"
        self.dict_metric = {}
        self.idx_csv = 1
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_size_va_history.csv')

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            self.measure.append(kwargs["valid_actions"].size(1))

    def compute_(self, **kwargs):
        self.metric = [np.round(np.mean(self.measure))]
        self.dict_metric[self.idx_csv] = self.measure
        self.idx_csv += 1

    def write_to_csv(self):
        size_per_timestep = [[val[i] for val in self.dict_metric.values()] for i in
                             range(min([len(val) for val in self.dict_metric.values()]))]
        self.dict_metric["mean_size"] = np.round(np.mean([np.mean(val) for val in self.dict_metric.values()]))
        self.dict_metric["mean_by_timestep"] = [np.round(np.mean(item)) for item in size_per_timestep]
        write_to_csv(self.out_csv_file, self.dict_metric)


class LMVAMetric(Metric):
    '''Monitor the mismatch between the valid actions space and the ref questions.'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.counter = 0
        self.key = "lm_valid_actions"

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            closest_question = self.agent.env.clevr_dataset.word2idx(kwargs["closest_question"].split())
            if len(closest_question) > self.idx_word:
                if closest_question[self.idx_word] not in kwargs["valid_actions"]:
                    self.counter += 1
                    logging.info("+VA")

    def compute_(self, **kwargs):
        self.metric = [self.counter]


class PoliciesRatioMetric(Metric):
    '''to monitor the discrepancy between the truncated policy (used for action selection) and the learned policy'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "policies_discrepancy"

    def fill_(self, **kwargs):
        ratios = np.exp(
            kwargs["log_probs"].detach().cpu().numpy() - kwargs["log_probs_truncated"].detach().cpu().numpy())
        self.measure.append(ratios)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class LMPolicyProbsRatio(Metric):
    '''to monitor the difference between the proba given by the lm for the words choosen and the probas given by the policy.'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "lm_policy_probs_ratio"

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            lm_log_probs = kwargs["actions_probs"][kwargs["valid_actions"] == kwargs["action"]].detach().cpu().numpy()
            ratios = np.exp(lm_log_probs - kwargs["log_probs"].detach().cpu().numpy())
        else:
            ratios = 0
        self.measure.append(ratios)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class ActionProbs(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "action_probs"

    def fill_(self, **kwargs):
        self.measure.append(kwargs["log_probs"])

    def compute_(self, **kwargs):
        ep_log_probs = torch.stack(self.measure).clone().detach()
        self.ep_probs = np.round(np.exp(ep_log_probs.cpu().squeeze().numpy()), decimals=5)
        self.metric.append(np.mean(self.ep_probs))

    def log(self, **kwargs):
        logging.info('episode action probs: {}'.format(self.ep_probs))


class ActionProbsTruncated(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "action_probs_truncated"

    def fill_(self, **kwargs):
        self.measure.append(kwargs["log_probs_truncated"])

    def compute_(self, **kwargs):
        ep_log_probs_truncated = torch.stack(self.measure).clone().detach()
        self.ep_probs_truncated = np.round(np.exp(ep_log_probs_truncated.cpu().squeeze().numpy()), decimals=5)
        self.metric.append(np.mean(self.ep_probs_truncated))

    def log(self, **kwargs):
        logging.info('episode action probs truncated: {}'.format(self.ep_probs_truncated))


class LMActionProbs(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "action_probs_lm"

    def fill_(self, **kwargs):
        if kwargs["action"] in kwargs["valid_actions"]:
            self.measure.append(kwargs["actions_probs"][kwargs["valid_actions"] == kwargs["action"]])
        else:
            self.measure.append(torch.tensor([0.]).to(self.agent.device))

    def compute_(self, **kwargs):
        lm_probs = torch.stack(self.measure).cpu().clone().detach()
        self.ep_lm_probs = np.round(lm_probs.cpu().squeeze().numpy(), decimals=5)
        self.metric.append(np.mean(self.ep_lm_probs))

    def log(self, **kwargs):
        logging.info('episode action probs from the LANGUAGE MODEL: {}'.format(self.ep_lm_probs))


class RunningReturn(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "running_return"
        self.running_return = 0
        self.idx_episode = 1
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_running_return_history.csv')

    def fill_(self, **kwargs):
        self.measure.append(kwargs["reward"])

    def compute_(self, **kwargs):
        ep_reward = np.sum(self.measure)
        self.running_return = 0.05 * ep_reward + (1 - 0.05) * self.running_return
        self.metric = [self.running_return]
        self.dict_metric[self.idx_episode] = self.running_return
        self.idx_episode += 1

    def write_to_csv(self):
        write_to_csv(self.out_csv_file, self.dict_metric)


class EpsilonTruncation(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "eps_truncation"
        self.counter = 0
        self.out_txt_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key + '.txt')
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key + '.csv')
        self.dict_metric = dict.fromkeys(["Episode", "Img_idx", "Action", "Policy_prob"])
        for key in list(self.dict_metric.keys()):
            self.dict_metric[key] = []

    def fill_(self, **kwargs):
        if self.agent.epsilon_truncated > 0:
            if kwargs["action"] not in kwargs["valid_actions"]:
                self.counter += 1  # TODO: a moving average instead ?
                action_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["action"].cpu().numpy(), ignored=[])
                action_prob = np.exp(kwargs["log_probs"].cpu().detach().numpy()).item()
                self.measure.append("Episode {} - Img {}/ Action: {}/ Policy prob: {:2.4f}".format(kwargs["i_episode"],
                                                                                                   self.agent.env.img_idx,
                                                                                                   action_decoded,
                                                                                                   action_prob))
                self.dict_metric["Episode"].append(kwargs["i_episode"])
                self.dict_metric["Img_idx"].append(self.agent.env.img_idx)
                self.dict_metric["Action"].append(action_decoded)
                self.dict_metric["Policy_prob"].append(np.round(action_prob, decimals=4))

    def write_to_csv(self):
        if self.agent.epsilon_truncated > 0:
            # write_to_csv(self.out_csv_file, self.dict_metric)
            write_to_csv_by_row(self.out_csv_file, self.dict_metric)

    def compute_(self, **kwargs):
        if self.agent.epsilon_truncated > 0:
            self.metric = [self.counter]
            string = '\n'.join(self.measure)
            with open(self.out_txt_file, 'a') as f:
                f.write(string + '\n')


# --------------------  TEST METRICS ----------------------------------------------------------------------------------------------------------------------------

class DialogMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "text"
        self.key = "dialog"
        self.out_dialog_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key + '.html')
        self.h5_dialog_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key + '.h5')
        self.generated_dialog = {}
        self.path_images = self.agent.env.path_images

    def fill_(self, **kwargs):
        pass

    def reinit_train_test(self, train_test):
        self.train_test = train_test
        self.out_dialog_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key + '.html')

    def compute_(self, **kwargs):
        with torch.no_grad():
            if not self.train_test + '_' + self.key in self.generated_dialog.keys():
                self.generated_dialog[self.train_test + '_' + self.key] = [kwargs["state"].text.squeeze().cpu()]
            else:
                self.generated_dialog[self.train_test + '_' + self.key].append(
                    kwargs["state"].text.cpu().view(-1))
            state_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["state"].text[:, 1:].numpy()[0],
                                                                  ignored=[])
            if self.agent.env.reward_type == 'vqa':
                pred_answer_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["pred_answer"].numpy(),
                                                                            decode_answers=True)
                ref_answer_decoded = self.agent.env.clevr_dataset.idx2word([kwargs["ref_answer"].numpy().item()],
                                                                           decode_answers=True)
                ref_question_decoded = kwargs["ref_questions_decoded"][kwargs["question_idx"]]

                values = [kwargs["img_idx"], kwargs["question_idx"], state_decoded, pred_answer_decoded,
                          ref_answer_decoded, ref_question_decoded]
            else:
                values = [kwargs["img_idx"], state_decoded, kwargs["closest_question"]]
            string = '<table><tr>'
            if self.train_test[:4] == "test":
                img_name = "CLEVR_{}_{:06d}.png".format(self.agent.env.clevr_mode, kwargs["img_idx"])
                path = os.path.join(self.agent.env.data_path, self.path_images, "images",self.agent.env.clevr_mode, img_name)
                values.append("<img src={}>".format(os.path.abspath(path)))

            string += "<td><ul><li>" + "</li><li>".join(list(map(str, values))) + "</li></ul></td></tr></table>"

            self.metric.append(string)
            # write dialog in a .html file:
            with open(self.out_dialog_file, 'a') as f:
                f.write(string + '\n')
            pass

    def write_to_csv(self):
        '''save padded array of generated dialog for later use (for example with word cloud)'''
        if self.train_test != "train":
            for key, dialog in self.generated_dialog.items():
                generated_dialog = pad_sequence(dialog, batch_first=True).cpu().numpy()
                with h5py.File(self.h5_dialog_file, 'w') as f:
                    f.create_dataset(key, data=generated_dialog)


class PPLMetric(Metric):
    """
    https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
    """

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "ppl"
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key)

    def fill_(self, **kwargs):
        if kwargs["test_mode"] == 'sampling':
            with torch.no_grad():
                if kwargs["done"]:
                    for ref_question in kwargs["ref_question"]:
                        inp_question = ref_question[:-1]
                        inp_question = torch.cat(
                            [torch.tensor(self.agent.env.special_tokens.SOS_idx).view(1), inp_question]).to(
                            self.agent.device)  # adding SOS token.
                        target_question = ref_question[1:]
                        target_question = torch.cat(
                            [target_question, torch.tensor(self.agent.env.special_tokens.EOS_idx).view(1)]).to(
                            self.agent.device)  # adding EOS token.
                        for i in range(len(inp_question)):
                            inputs = inp_question[:i + 1].unsqueeze(0)
                            policy_dist, policy_dist_truncated, _ = self.agent.policy(inputs, kwargs["state"].img,
                                                                                      valid_actions=kwargs[
                                                                                          "valid_actions"],
                                                                                      state_answer=self.agent.env.ref_answer)  # TODO: bug here with vqa task.
                            log_prob = policy_dist_truncated.log_prob(target_question[i])
                            self.measure.append(log_prob)

    def compute_(self, **kwargs):
        if kwargs["test_mode"] == 'sampling':
            ppl = torch.exp(-torch.stack(self.measure).sum() / len(self.measure)).cpu().numpy().item()
            self.metric.append(ppl)
            if not self.train_test + '_' + self.key in self.dict_metric:
                self.dict_metric[self.train_test + '_' + self.key] = [self.metric[-1]]
            else:
                self.dict_metric[self.train_test + '_' + self.key].append(self.metric[-1])

    def write(self):
        pass


class PPLDialogfromLM(Metric):
    '''Computes the PPL of the Language Model over the generated dialog'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "ppl_dialog_lm"
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key)

    def fill_(self, **kwargs):
        if kwargs["done"]:
            with torch.no_grad():
                log_probas, hidden = self.agent.pretrained_lm(kwargs["new_state"].text[:, :-1].to(self.agent.device))
                for i, word in enumerate(kwargs["new_state"].text[:, 1:].view(-1)):
                    self.measure.append(log_probas[i, word.cpu().numpy()])

    def compute_(self, **kwargs):
        ppl = torch.exp(-torch.stack(self.measure).sum() / len(self.measure)).cpu().numpy().item()
        self.metric.append(ppl)
        if not self.train_test + '_' + self.key in self.dict_metric:
            self.dict_metric[self.train_test + '_' + self.key] = [self.metric[-1]]
        else:
            self.dict_metric[self.train_test + '_' + self.key].append(self.metric[-1])

    def write(self):
        pass


class Return(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "return"
        self.idx_episode = 1
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_return_history.csv')

    def fill_(self, **kwargs):
        self.measure.append(kwargs["reward"])

    def compute_(self, **kwargs):
        ep_return = np.sum(self.measure)
        self.metric = [ep_return]
        self.dict_metric[self.idx_episode] = ep_return
        self.idx_episode += 1

    def write_to_csv(self):
        write_to_csv(self.out_csv_file, self.dict_metric)

    def post_treatment(self):
        csv = os.path.join(self.agent.out_path, self.train_test + '_std_history.csv')
        serie = pd.Series(self.dict_metric).rolling(window=100).std()
        serie.to_csv(csv)


class Reward2Metric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "reward"

    def fill_(self, **kwargs):
        self.measure.append(kwargs["reward"])

    def compute_(self, **kwargs):
        self.metric.append(np.sum(self.measure))


class RewardMetric(Metric):
    """Computes:
    - The raw reward (the one used in the training algo)
    - The normalised reward (the one monitored at test time)
    - The length of each test episode
    """

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "reward"
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key)
        self.measure = {}

    def fill_(self, **kwargs):
        condition = kwargs["done"] if self.agent.env.reward_func.type == "episode" else True
        if condition:
            self.measure["reward"] = [kwargs["reward"]]
            len_episode = len(self.agent.env.clevr_dataset.idx2word(kwargs["new_state"].text[0, 1:].cpu().numpy(),
                                                                    ignored=[]).split())
            if self.agent.env.reward_type == 'levenshtein_':
                norm_reward = [kwargs["reward"] / max(len_episode,
                                                      len(kwargs["closest_question"].split()))]
            else:
                norm_reward = self.measure["reward"]
            self.measure["norm_reward"] = norm_reward
            self.measure["len_dialog"] = [len_episode]

    def compute_(self, **kwargs):
        if not self.train_test + '_' + self.key in self.dict_metric:
            self.dict_metric[self.train_test + '_' + self.key] = self.measure
        else:
            for key, value in self.measure.items():
                self.dict_metric[self.train_test + '_' + self.key][key].append(value[-1])

    def compute(self, **kwargs):
        self.compute_(**kwargs)
        self.measure = {}
        self.idx_word = 0
        self.idx_step = 0

    def write_to_csv(self):
        for key, value in self.dict_metric.items():
            self.dict_stats[key] = {}
            for k, v in value.items():
                self.dict_stats[key][k] = [np.round(np.mean(v), decimals=3), np.round(np.std(v), decimals=3), len(v)]
                logging.info('{} mean: {}'.format(key + '___' + k, np.round(np.mean(v), decimals=3)))
                logging.info('{} std: {}'.format(key + '___' + k, np.round(np.mean(v), decimals=3)))
        write_to_csv(self.out_csv_file + '_stats.csv', self.dict_stats)

    def write(self):
        '''Overwrite write function to avoid logging on tensorboard.'''
        pass


class BleuMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "bleu"
        self.train_test = train_test
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key)

    def fill_(self, **kwargs):
        if kwargs["done"]:
            question_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["state"].text.numpy()[0], ignored=["<SOS>"],
                                                                     stop_at_end=True)
            ref_questions = kwargs["ref_questions_decoded"]
            ref_questions = [q.split() for q in ref_questions]
            question_tokens = question_decoded.split()
            score = sentence_bleu(ref_questions, question_tokens)
            self.measure.append(score)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))
        if not self.train_test + '_' + self.key in self.dict_metric:
            self.dict_metric[self.train_test + '_' + self.key] = [self.metric[-1]]
        else:
            self.dict_metric[self.train_test + '_' + self.key].append(self.metric[-1])

    def write(self):
        '''Overwrite write function to avoid logging on tensorboard.'''
        pass


# ------------------------ DIVERSITY METRICS -------------------------------------------------------------------------------------------------------------------

class RefQuestionsMetric(Metric):
    '''
    Compute the ratio of Unique closest questions on all the set of questions generated for the same image.
    '''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "ratio_closest_questions"
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key)

    def fill_(self, **kwargs):
        if "test_images" in self.train_test:
            if kwargs["done"]:
                self.measure.append(kwargs["closest_question"])

    def compute_(self, **kwargs):
        if "test_images" and "sampling" in self.train_test:
            if len(self.measure) == kwargs["ref_question"].size(0):
                unique_ratio = len(list(set(self.measure))) / len(self.measure)
                self.metric.append(unique_ratio)
                self.measure = []
                if not self.train_test + '_' + self.key in self.dict_metric:
                    self.dict_metric[self.train_test + '_' + self.key] = [self.metric[-1]]
                else:
                    self.dict_metric[self.train_test + '_' + self.key].append(self.metric[-1])

    def compute(self, **kwargs):
        self.compute_(**kwargs)
        self.idx_word = 0
        self.idx_step = 0

    def write(self):
        pass


class TTRQuestionMetric(Metric):
    '''
    Compute the token-to-token ratio for each question (useful to measure language drift).
    '''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "ttr_question"
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key)

    def fill_(self, **kwargs):
        if kwargs["done"]:
            self.measure = kwargs["new_state"].text.numpy()[0]

    def compute_(self, **kwargs):
        diversity_metric = len(set(list(self.measure))) / len(self.measure)
        self.metric.append(diversity_metric)
        if not self.train_test + '_' + self.key in self.dict_metric:
            self.dict_metric[self.train_test + '_' + self.key] = [self.metric[-1]]
        else:
            self.dict_metric[self.train_test + '_' + self.key].append(self.metric[-1])

    def write(self):
        pass


class UniqueWordsMetric(Metric):
    '''Compute the ratio of Unique Words for the set of questions generated for each image. Allows to measure vocabulary diversity.'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "unique_words"
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key)

    def fill_(self, **kwargs):
        if "sampling" in self.train_test:
            if kwargs["done"]:
                self.measure.append(list(kwargs["new_state"].text.numpy()[0]))

    def compute_(self, **kwargs):
        if "sampling" in self.train_test:
            if len(self.measure) == kwargs["ref_question"].size(0):
                arr = np.array(self.measure).flatten()
                unique_tokens = np.unique(arr)
                diversity_metric = len(unique_tokens) / len(arr)
                self.metric.append(diversity_metric)
                self.measure = []
                if not self.train_test + '_' + self.key in self.dict_metric:
                    self.dict_metric[self.train_test + '_' + self.key] = [self.metric[-1]]
                else:
                    self.dict_metric[self.train_test + '_' + self.key].append(self.metric[-1])

    def compute(self, **kwargs):
        self.compute_(**kwargs)
        self.idx_word = 0
        self.idx_step = 0

    def write(self):
        pass


# --------------------------------------- OTHERS ----------------------------------------------------------------------------------------------------

class PolicyMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "text"
        self.key = "policy"

    def fill_(self, **kwargs):
        # compute top_k_words from the Policy:
        with torch.no_grad():
            state_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["state"].text.numpy()[0], ignored=[])
            top_k_weights, top_k_indices = torch.topk(kwargs["dist"].probs, 5, sorted=True)
            top_words_decoded = self.agent.env.clevr_dataset.idx2word(top_k_indices.cpu().numpy()[0])
            # get top_words from the language model:
            seq_len = kwargs["state"].text.size(1)
            log_probas, _ = self.agent.pretrained_lm(kwargs["state"].text.to(self.agent.device))
            log_probas = log_probas.view(len(kwargs["state"].text), seq_len, -1)
            _, top_k_indices_lm = torch.topk(log_probas[:, -1, :], 10, sorted=True)
            top_k_indices, top_k_weights, top_k_indices_lm = top_k_indices.squeeze(), top_k_weights.squeeze(), top_k_indices_lm.squeeze()
            in_top_k_words_lm = []
            for i in top_k_indices:
                if i in top_k_indices_lm:
                    in_top_k_words_lm.append("Y")
                else:
                    in_top_k_words_lm.append("N")
            weights_words = ["{}/{:.3f}/{}".format(word, weight, top_k_lm, number=3) for word, weight, top_k_lm in
                             zip(top_words_decoded.split(), top_k_weights.cpu().detach().numpy(), in_top_k_words_lm)]
            self.measure.append("next possible words for {} : {}".format(state_decoded, ", ".join(weights_words)))

    def compute_(self, **kwargs):
        self.metric = self.measure

    def write(self):
        pass

    def log(self, **kwargs):
        logging.info('---------------------Policy Top Words------------------------------')
        logging.info('\n'.join(self.metric))
        logging.info('--------------------------------------------------------------------')


metrics = {"dialog": DialogMetric, "valid_actions": VAMetric, "lm_valid_actions": LMVAMetric, "reward": RewardMetric,
           "policies_discrepancy": PoliciesRatioMetric, "lm_policy_probs_ratio": LMPolicyProbsRatio, "bleu": BleuMetric,
           "ppl": PPLMetric, "ppl_dialog_lm": PPLDialogfromLM, "size_valid_actions": SizeVAMetric,
           "ttr_question": TTRQuestionMetric, "unique_words": UniqueWordsMetric,
           "ratio_closest_questions": RefQuestionsMetric,
           "action_probs": ActionProbs, "action_probs_truncated": ActionProbsTruncated,
           "action_probs_lm": LMActionProbs,
           "running_return": RunningReturn,
           "policy": PolicyMetric,
           "eps_truncation": EpsilonTruncation, "return": Return}
