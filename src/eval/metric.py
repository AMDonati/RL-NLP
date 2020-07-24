import numpy as np
import torch
import time
import logging


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

    def write(self, **kwargs):
        if self.type == "scalar":
            self.agent.writer.add_scalar(self.train_test + "_" + self.key, np.mean(self.metric), self.idx_write)
        else:
            self.agent.writer.add_text(self.train_test + "_" + self.key, '  \n'.join(self.metric[-1:]), self.idx_write)
        self.idx_write += 1
        self.metric_history.extend(self.metric)
        self.metric = []


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
                             zip(top_words_decoded.split(), kwargs["actions_probs"].cpu().detach().exp().numpy()[0])]
            string = "next possible words for {} : {}".format(state_decoded, ", ".join(weights_words))
        else:
            string = ""
        self.measure.append(string)

    def compute_(self, **kwargs):
        self.metric = self.measure
        pass


class DialogMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "text"
        self.key = "dialog"

    def fill_(self, **kwargs):
        pass

    def compute_(self, **kwargs):
        state_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["state"].text[:, 1:].numpy()[0])
        closest_question_decoded = kwargs["closest_question"]
        self.metric.append(state_decoded + '---closest question---' + closest_question_decoded)
        pass


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


class PPLMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "ppl"

    def fill_(self, **kwargs):
        if self.agent.pretrained_lm is None:
            target_word_log_prob = kwargs["dist"].log_prob(kwargs["ref_question"][self.idx_word].to(self.agent.device))
            self.measure.append(target_word_log_prob)
        else:
            if kwargs["ref_question"].to(self.agent.device) in kwargs["valid_actions"]:
                target_word = list(kwargs["valid_actions"].view(-1).cpu().numpy()).index(
                    kwargs["ref_question"][self.idx_word])
                target_word_log_prob = kwargs["dist"].log_prob(
                    torch.tensor([target_word]).float().to(self.agent.device))
                self.measure.append(target_word_log_prob)
            # else:
            # case where the target word is not in the top words of the language model
            # target_word_log_prob = torch.tensor([-10]).float().to(self.device) #TODO: remove the else.
        # self.idx_word += 1
        # self.idx_step += 1

    def compute_(self, **kwargs):
        ppl = torch.exp(-torch.stack(self.measure).sum() / self.idx_step).detach().numpy()
        self.metric.append(ppl)
        self.reset()


class RewardMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = "reward"

    def fill_(self, **kwargs):
        condition = kwargs["done"] if self.agent.env.reward_func.type == "episode" else True
        if condition:
            self.measure.append(kwargs["reward"])

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class LMVAMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.counter = 0
        self.key = "invalid_actions"

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


metrics = {"dialog": DialogMetric, "valid_actions": VAMetric, "lm_valid_actions": LMVAMetric, "reward": RewardMetric,
           "policies_discrepancy": PoliciesRatioMetric, "lm_policy_probs_ratio": LMPolicyProbsRatio}

# TODO: add TTR metric, BLEU score.
