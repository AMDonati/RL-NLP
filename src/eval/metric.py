import numpy as np
import torch


class Metric:
    def __init__(self, agent, train_test):
        self.measure = []
        self.metric = []
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
        self.compute_()
        self.measure = []
        self.idx_word = 0
        self.idx_step = 0

    def reset(self):
        self.idx_word = 0

    def write(self, **kwargs):
        if self.type == "scalar":
            self.agent.writer.add_scalar(self.key, np.mean(self.metric), self.idx_write)
        else:
            self.agent.writer.add_text(self.key, '  \n'.join(self.metric), self.idx_write)
        self.idx_write += 1
        self.metric = []


class LMMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "text"
        self.key = train_test + "_" + "lm"

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
        self.key = train_test + "_" + "valid_actions"

    def fill_(self, **kwargs):
        state_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["state"].text[:, :-1].numpy()[0])
        if kwargs["valid_actions"] is not None:
            top_words_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["valid_actions"].cpu().numpy()[0])
            weights_words = ["{}/{:.3f}".format(word, weight, number=3) for word, weight in
                             zip(top_words_decoded.split(), kwargs["actions_probs"].cpu().detach().exp().numpy()[0])]
            string = "next possible words for {} : {}".format(state_decoded, ", ".join(weights_words))
        else:
            string = ""
        ref_questions = [w + ' <EOS>' for w in kwargs["ref_question"]]
        target_words = [w.split()[self.idx_word] for w in ref_questions]
        string = string + '--- target words: {}'.format(', '.join(target_words)) + '--- true action: {}'.format(
            self.agent.env.clevr_dataset.idx2word(kwargs["state"].text[:, -1].numpy()))
        self.measure.append(string)

    # self.idx_word += 1

    def compute_(self, **kwargs):
        self.metric = [self.measure[-1]]
        pass


class DialogMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "text"
        self.key = train_test + "_" + "dialog"

    def fill_(self, **kwargs):
        if kwargs["done"]:
            state_decoded = self.agent.env.clevr_dataset.idx2word(kwargs["state"].text.numpy()[0])
            closest_question_decoded = kwargs["closest_question"]
            self.measure.append(state_decoded + '---closest question---' + closest_question_decoded)

    def compute_(self, **kwargs):
        self.metric = [self.measure[-1]]
        pass


class RefQuestionsMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.key = train_test + "_" + "ratio_closest_questions"

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
        self.key = train_test + "_" + "ppl"

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
        self.key = train_test + "_" + "reward"

    def fill_(self, **kwargs):
        self.idx_word += 1
        self.measure.append(kwargs["reward"])

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class LMVAMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test)
        self.type = "scalar"
        self.counter = 0
        self.key = train_test + "_" + "invalid_actions"

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            closest_question = self.agent.env.clevr_dataset.word2idx(kwargs["closest_question"].split())
            if closest_question[self.idx_word] not in kwargs["valid_actions"]:
                self.counter += 1

    def compute_(self, **kwargs):
        self.metric = [self.counter]
