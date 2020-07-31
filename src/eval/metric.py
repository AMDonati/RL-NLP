import numpy as np
import torch
import logging
from nltk.translate.bleu_score import sentence_bleu
import os
from utils.utils_train import write_to_csv

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
    def write_to_csv(self):
        pass

# ----------------------------------    TRAIN METRICS -------------------------------------------------------------------------------------

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

class LMVAMetric(Metric):
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


# --------------------  TEST METRICS ----------------------------------------------------------------------------------------------------------------------------

class DialogMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "text"
        self.key = "dialog"
        self.out_dialog_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key + '.txt')

    def fill_(self, **kwargs):
        pass

    def reinit_train_test(self, train_test):
        self.train_test = train_test
        self.out_dialog_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key + '.txt')

    def compute_(self, **kwargs):
        state_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["state"].text[:, 1:].numpy()[0])
        closest_question_decoded = kwargs["closest_question"]
        string = state_decoded + '---closest question---' + closest_question_decoded
        self.metric.append(string)
        # write dialog in a .txt file:
        with open(self.out_dialog_file, 'a') as f:
            f.write(string + '\n')
        pass

class PPLMetric(Metric):
    """
    https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
    """
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "ppl"
        self.dict_ppl = {}
        self.dict_stats = {}
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key)

    def fill_(self, **kwargs):
        if kwargs["done"]:
            for ref_question in kwargs["ref_question"]: #TODO: add SOS and EOS token.
                inp_question = ref_question[:-1]
                inp_question = torch.cat([torch.tensor(self.agent.env.special_tokens.SOS_idx).view(1), inp_question]).to(self.agent.device) # adding SOS token.
                target_question = ref_question[1:]
                target_question = torch.cat([target_question, torch.tensor(self.agent.env.special_tokens.EOS_idx).view(1)]).to(self.agent.device)
                for i in range(len(inp_question)):
                    inputs = inp_question[:i + 1].unsqueeze(0)
                    policy_dist, policy_dist_truncated, _ = self.agent.policy(inputs, kwargs["state"].img, valid_actions=kwargs["valid_actions"])
                    log_prob = policy_dist_truncated.log_prob(target_question[i])
                    self.measure.append(log_prob)

    def compute_(self, **kwargs):
        ppl = torch.exp(-torch.stack(self.measure).sum() / len(self.measure)).detach().cpu().numpy().item()
        self.metric.append(ppl)
        if not self.train_test + '_' + self.key in self.dict_ppl:
            self.dict_ppl[self.train_test + '_' + self.key] = [self.metric[-1]]
        else:
            self.dict_ppl[self.train_test + '_' + self.key].append(self.metric[-1])

    def write_to_csv(self):
        for key, value in self.dict_ppl.items():
            self.dict_stats[key] = [np.mean(value), np.std(value), len(value)]
            logging.info('{} mean: {}'.format(key, np.mean(value)))
            logging.info('{} std: {}'.format(key, np.std(value)))
            self.dict_ppl[key].append(np.mean(value))
            self.dict_ppl[key].append(np.std(value))
        write_to_csv(self.out_csv_file + '.csv', self.dict_ppl)
        write_to_csv(self.out_csv_file + '_stats.csv', self.dict_stats)

    def write(self):
        pass

class PPLDialogfromLM(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "ppl_dialog_lm"
        self.dict_ppl, self.dict_stats = {}, {}
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key)

    def fill_(self, **kwargs):
        if kwargs["done"]:
            with torch.no_grad():
                log_probas, hidden = self.agent.pretrained_lm(kwargs["new_state"].text[:,:-1].to(self.agent.device))
                for i, word in enumerate(kwargs["new_state"].text[:,1:].view(-1)):
                    self.measure.append(log_probas[i,word.cpu().numpy()])

    def compute_(self, **kwargs):
        ppl = torch.exp(-torch.stack(self.measure).sum() / len(self.measure)).cpu().numpy().item()
        self.metric.append(ppl)
        if not self.train_test + '_' + self.key in self.dict_ppl:
            self.dict_ppl[self.train_test + '_' + self.key] = [self.metric[-1]]
        else:
            self.dict_ppl[self.train_test + '_' + self.key].append(self.metric[-1])

    def write_to_csv(self):
        for key, value in self.dict_ppl.items():
            self.dict_stats[key] = [np.mean(value), np.std(value), len(value)]
            logging.info('{} mean: {}'.format(key, np.mean(value)))
            logging.info('{} std: {}'.format(key, np.std(value)))
            self.dict_ppl[key].append(np.mean(value))
            self.dict_ppl[key].append(np.std(value))
        write_to_csv(self.out_csv_file + '.csv', self.dict_ppl)
        write_to_csv(self.out_csv_file + '_stats.csv', self.dict_stats)

    def write(self):
        pass


class RewardMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "reward"
        self.out_csv_file = os.path.join(self.agent.out_path, self.train_test + '_' + self.key)
        self.measure = {}
        self.dict_rewards, self.dict_stats = {}, {}

    def fill_(self, **kwargs):
        condition = kwargs["done"] if self.agent.env.reward_func.type == "episode" else True
        if condition:
            self.measure["reward"] = [kwargs["reward"]] #TODO: does not work with differential reward ?
            norm_reward = [kwargs["reward"]/max(kwargs["new_state"].text[:,1:].squeeze().cpu().numpy().size, len(kwargs["closest_question"].split()))]
            self.measure["norm_reward"] = norm_reward
            self.measure["len_dialog"] = [kwargs["new_state"].text[:,1:].squeeze().cpu().numpy().size]

    def compute_(self, **kwargs):
        if not self.train_test + '_' + self.key in self.dict_rewards:
            self.dict_rewards[self.train_test + '_' + self.key] = self.measure
        else:
            for key, value in self.measure.items():
                self.dict_rewards[self.train_test + '_' + self.key][key].append(value[-1])

    def compute(self, **kwargs):
        self.compute_(**kwargs)
        self.measure = {}
        self.idx_word = 0
        self.idx_step = 0

    def write_to_csv(self):
        for key, value in self.dict_rewards.items():
            self.dict_stats[key] = {}
            for k, v in value.items():
                self.dict_stats[key][k] = [np.mean(v), np.std(v), len(v)]
                logging.info('{} mean: {}'.format(key + '___' + k, np.mean(v)))
                logging.info('{} std: {}'.format(key + '___' + k, np.std(v)))
                self.dict_rewards[key][k].append(np.mean(v))
                self.dict_rewards[key][k].append(np.std(v))
        write_to_csv(self.out_csv_file+'.csv', self.dict_rewards)
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
        self.dict_bleus, self.dict_stats = {}, {}

    def fill_(self, **kwargs):
        if kwargs["done"]:
            question_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["state"].text.numpy()[0], ignored=["<SOS>"],
                                                                     stop_at_end=True)
            ref_questions = [q.split() for q in self.agent.env.ref_questions_decoded]
            question_tokens = question_decoded.split()
            score = sentence_bleu(ref_questions, question_tokens)
            self.measure.append(score)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))
        if not self.train_test + '_' + self.key in self.dict_bleus:
            self.dict_bleus[self.train_test + '_' + self.key] = [self.metric[-1]]
        else:
            self.dict_bleus[self.train_test + '_' + self.key].append(self.metric[-1])

    def write_to_csv(self):
        for key, value in self.dict_bleus.items():
            self.dict_stats[key] = [np.mean(value), np.std(value), len(value)]
            logging.info('{} mean: {}'.format(key, np.mean(value)))
            logging.info('{} std: {}'.format(key, np.std(value)))
            self.dict_bleus[key].append(np.mean(value))
            self.dict_bleus[key].append(np.std(value))
        write_to_csv(self.out_csv_file+'.csv', self.dict_bleus)
        write_to_csv(self.out_csv_file + '_stats.csv', self.dict_stats)

    def write(self):
        '''Overwrite write function to avoid logging on tensorboard.'''
        pass

# ------------------------ DIVERSITY METRICS -------------------------------------------------------------------------------------------------------------------

class RefQuestionsMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "ratio_closest_questions"

    def fill_(self, **kwargs):
        if kwargs["done"]:
            self.measure.append(kwargs["closest_question"])

    def compute_(self, **kwargs):
        unique_ratio = len(list(set(self.measure))) / len(self.measure)
        self.metric.append(unique_ratio)


class TTRMetric(Metric):
    def __init__(self, agent):
        Metric.__init__(self, agent)
        self.type = "scalar"
        self.key = "ttr"

    def fill_(self, **kwargs):
        self.measure.append(kwargs["state"].text.numpy()[0])

    def compute_(self, **kwargs):
        last_text = [item for sublist in self.measure[-min(10, len(self.measure)):] for item in sublist]
        diversity_metric = len(set(last_text)) / len(last_text)
        self.metric.append(diversity_metric)

# --------------------------------------- OTHERS ----------------------------------------------------------------------------------------------------

class LMMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "text"
        self.key = "lm"

    def fill_(self, **kwargs):
        state_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["state"].text.numpy()[0])
        top_k_weights, top_k_indices = torch.topk(kwargs["dist"].probs, self.agent.num_truncated, sorted=True)
        if self.agent.pretrained_lm != None:
            top_k_indices = torch.gather(kwargs["valid_actions"], 1, top_k_indices)
        top_words_decoded = self.agent.env.clevr_dataset.idx2word(top_k_indices.cpu().numpy()[0])
        weights_words = ["{}/{:.3f}".format(word, weight, number=3) for word, weight in
                         zip(top_words_decoded.split(), top_k_weights[0].cpu().detach().numpy())]
        self.measure.append("next possible words for {} : {}".format(state_decoded, ", ".join(weights_words)))

    def compute_(self, **kwargs):
        self.metric = [self.measure[-1]]
        pass


metrics = {"dialog": DialogMetric, "valid_actions": VAMetric, "lm_valid_actions": LMVAMetric, "reward": RewardMetric,
           "policies_discrepancy": PoliciesRatioMetric, "lm_policy_probs_ratio": LMPolicyProbsRatio, "bleu": BleuMetric, "ppl": PPLMetric, "ppl_dialog_lm": PPLDialogfromLM}

