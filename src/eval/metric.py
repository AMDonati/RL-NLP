import heapq
import logging
import operator
import os
os.environ['TRANSFORMERS_CACHE'] = "/cache"

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from nlgeval.pycocoevalcap.cider.cider import Cider
from nltk.translate.meteor_score import meteor_score
#import nltk
from tools.refer.evaluation.tokenizer.ptbtokenizer import PTBTokenizer
from torch.nn.utils.rnn import pad_sequence
from transformers import OpenAIGPTTokenizer, OpenAIGPTLMHeadModel
from transformers import AutoModelWithLMHead, AutoTokenizer

from RL_toolbox.reward import rewards
# If modifying these scopes, delete the file token.pickle.
from data_provider.CLEVR_Dataset import CLEVR_Dataset
from models.language_model import ClevrLanguageModel

#nltk.download('wordnet')

SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']

logger = logging.getLogger()


def _strip(s):
    return s.strip()


class Metric:
    def __init__(self, agent, train_test, key, type, env_mode, trunc, sampling):
        self.measure = []
        self.metric = []
        self.metric_history = []
        self.metric_diversity = []
        self.metric_diversity_history = []
        self.idxs_to_select = []
        self.idx_to_select = None
        self.idx_step = 0
        self.idx_word = 0
        self.idx_compute = 0
        self.idx_write = 1
        self.dataset = agent.env.dataset
        self.out_path = agent.out_path
        self.writer = agent.writer
        self.language_model = agent.truncation.language_model
        self.policy = agent.policy
        self.reward_type = agent.env.reward_type
        self.type = type
        self.key = key
        self.train_test = train_test
        self.id = "_".join([env_mode, trunc, sampling])
        self.env_mode = env_mode
        self.trunc = trunc
        self.sampling = sampling
        # self.dict_metric, self.dict_stats = {}, {}  # for csv writing.
        self.name = train_test + "_" + self.id + '_' + self.key
        self.out_csv_file = os.path.join(self.out_path, "metrics", self.name + ".csv")
        self.out_div_csv_file = os.path.join(self.out_path, "diversity", self.name + ".csv")
        self.stats = None
        self.stats_div = None
        self.to_tensorboard = True if key in metrics_to_tensorboard else False

    def fill(self, **kwargs):
        self.fill_(**kwargs)
        self.idx_word += 1
        self.idx_step += 1

    def compute(self, **kwargs):
        self.compute_(**kwargs)
        self.measure = []
        self.idx_word = 0
        self.idx_step = 0
        self.idx_compute += 1

    def reset(self):
        self.idx_word = 0

    def write(self, idx_to_select=False):
        if self.to_tensorboard:
            if self.type == "scalar":
                self.writer.add_scalar(self.name, np.mean(self.metric), self.idx_write)
            elif self.type == "text":
                self.writer.add_text(self.name, '  \n'.join(self.metric[-1:]), self.idx_write)
        self.idx_write += 1
        self.metric_history.extend(self.metric)
        if self.sampling != "greedy":
            self.metric_diversity.extend(self.metric)
        self.metric = []
        if idx_to_select and self.sampling == "sampling_ranking_lm":
            self.idx_to_select = self.idx_compute - 1

    def write_div(self, **kwargs):
        if self.type == "scalar" and self.metric_diversity:
            metric_diversity = [np.mean(self.metric_diversity), np.std(self.metric_diversity),
                                np.max(self.metric_diversity), np.min(self.metric_diversity)]
            self.metric_diversity_history.append(metric_diversity)
            self.metric_diversity = []
        if self.idx_to_select is not None and self.sampling == "sampling_ranking_lm":
            self.idxs_to_select.append(self.idx_to_select)

    def log(self, **kwargs):
        pass

    def get_stats(self, serie):
        return [serie.mean(), serie.std(), serie.size]

    def get_stats_div(self, df):
        return df.mean().to_dict()

    def post_treatment_(self):
        pass

    def post_treatment(self, num_episodes):
        if len(self.idxs_to_select) > 0 and self.sampling == "sampling_ranking_lm" and len(
                self.metric_history) == num_episodes * 10:
            self.metric_history = np.array(self.metric_history)
            self.metric_history = list(self.metric_history[self.idxs_to_select])
        self.post_treatment_()
        serie = pd.Series(self.metric_history)
        serie.to_csv(self.out_csv_file, index=False, header=False)

        if self.type == "scalar":
            self.stats = self.get_stats(serie)
            if self.metric_diversity_history:
                df = pd.DataFrame(data=self.metric_diversity_history, columns=["mean", "std", "max", "min"])
                df.to_csv(self.out_div_csv_file)
                self.stats_div = self.get_stats_div(df)


# ----------------------------------  TRAIN METRICS -------------------------------------------------------------------------------------

class VAMetric(Metric):
    '''Display the valid action space in the training log.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "valid_actions", "text", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        state_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text.numpy()[0],
                                                               ignored=['<PAD>'])
        if self.language_model.init_text is not None:
            state_decoded = self.language_model.init_text_short + "\n" + state_decoded
        string = ""
        if kwargs["valid_actions"] is not None:
            top_words_decoded = [self.dataset.question_tokenizer.decode([va]) for va in
                                 kwargs["valid_actions"].cpu().numpy()[0]]
            weights_words = ["{}/{:.3f}".format(word, weight, number=3) for word, weight in
                             zip(top_words_decoded, kwargs["actions_probs"].cpu().detach().numpy()[0])]
            string = "next possible words for {} : {}".format(state_decoded, ", ".join(weights_words))
        self.measure.append(string)

    def compute_(self, **kwargs):
        self.metric = self.measure

    def log(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            logger.info('---------------------Valid action space------------------------------')
            logger.info('\n'.join(self.metric))
            logger.info('---------------------------------------------------------------------')

    def write(self):
        pass


class SizeVAMetric(Metric):
    '''Compute the average size of the truncated action space during training for truncation functions proba_thr & sample_va'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "size_valid_actions", "scalar", env_mode, trunc, sampling)
        self.counter = 0

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            self.measure.append(kwargs["valid_actions"].size(1))

    def compute_(self, **kwargs):
        self.metric.extend(self.measure)


class SumProbsOverTruncated(Metric):
    '''Compute the sum of the probabilities the action space given by the language model.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "sum_probs_truncated", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        sum_over_truncated_space = 1
        if kwargs["valid_actions"] is not None:
            sum_over_truncated_space = torch.gather(kwargs["dist"].probs, -1,
                                                    kwargs["valid_actions"]).sum().cpu().detach().numpy()
        self.measure.append(float(sum_over_truncated_space))

    def compute_(self, **kwargs):
        self.metric.extend(self.measure)


# --------------------  TEST METRICS ----------------------------------------------------------------------------------------------------------------------------
class DialogMetric(Metric):
    """Display the test dialog."""

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "dialog", "text", env_mode, trunc, sampling)
        # self.out_dialog_file = os.path.join(self.out_path, self.train_test + '_' + self.key + '.txt')
        # self.h5_dialog_file = os.path.join(self.out_path, self.train_test + '_' + self.key + '.h5')
        self.generated_dialog = {}

    def fill_(self, **kwargs):
        pass

    def compute_(self, **kwargs):
        with torch.no_grad():
            state_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text[:, :].numpy()[0],
                                                                   ignored=[])
            if self.reward_type == 'vqa' or self.reward_type == "vilbert" or self.reward_type == "vilbert_rank2":
                pred_answer = [int(kwargs["pred_answer"].squeeze().detach().numpy())]
                pred_answer_decoded = self.dataset.answer_tokenizer.decode(text=pred_answer)
                ref_answer_decoded = self.dataset.answer_tokenizer.decode(kwargs["ref_answer"].view(1).numpy())
                ref_question_decoded = kwargs["ref_questions_decoded"]
                string = ' IMG {} - question index {}:'.format(kwargs["img_idx"],
                                                               kwargs[
                                                                   "question_idx"]) + '\n' + 'DIALOG:' + state_decoded + '\n' + 'REF QUESTION:' + \
                         ref_question_decoded[
                             0] + '\n' + 'VQA ANSWER:' + pred_answer_decoded + '\n' + 'TRUE ANSWER:' + ref_answer_decoded + '\n' + '-' * 40
            else:
                closest_question_decoded = kwargs["closest_question"]
                string = 'IMG {}:'.format(kwargs[
                                              "img_idx"]) + state_decoded + '\n' + 'CLOSEST QUESTION:' + closest_question_decoded + '\n' + '-' * 40

        self.metric.append(string)

    def log(self, **kwargs):
        if len(self.metric) > 0:
            logger.info(self.metric[-1])
            logger.info("-" * 40)

    def write_to_csv(self):
        '''save padded array of generated dialog for later use (for example with word cloud)'''
        if self.train_test != "train":
            for key, dialog in self.generated_dialog.items():
                generated_dialog = pad_sequence(dialog, batch_first=True).cpu().numpy()
                with h5py.File(self.h5_dialog_file, 'w') as f:
                    f.create_dataset(key, data=generated_dialog)


class DialogImageMetric(Metric):
    '''Display the Dialog on a html format at test time.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "dialog_image", "text", env_mode, trunc, sampling)
        self.generated_dialog = []
        self.condition_answer = agent.policy.condition_answer
        self.mode = agent.env.mode
        image_id_file = "clevr" if self.dataset.__class__ == CLEVR_Dataset else "coco"
        image_id_file = os.path.join("data", "drive", image_id_file + ".csv")
        self.list_image_ids = pd.read_csv(image_id_file, index_col="id_image")
        self.out_html_file = os.path.join(self.out_path, "metrics", self.name + ".html")
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            true_action = kwargs["ref_question"].view(-1)[kwargs["timestep"]]
            in_va = true_action.cpu() in kwargs["valid_actions"].cpu()
            _, indices = torch.sort(kwargs["log_probas_lm"], descending=True)
            rank = int(torch.nonzero(indices.squeeze().cpu() == true_action).squeeze().numpy())

            self.measure.append([in_va, rank])

    def compute_(self, **kwargs):
        with torch.no_grad():
            state_decoded = self.dataset.question_tokenizer.decode(text=kwargs["state"].text[:, 1:].numpy()[0],
                                                                   ignored=[])
            values = {}
            values["img"] = kwargs["img_idx"]
            values["question"] = state_decoded
            values["reward"] = round(kwargs["reward"], 3)
            values["ref_questions"] = kwargs["ref_questions_decoded"]
            values["in_valid_actions"] = self.measure

            if self.condition_answer != "none":
                ref_answer_decoded = self.dataset.answer_tokenizer.decode(kwargs["ref_answer"].view(1).numpy())
                values["ref_answer"] = ref_answer_decoded

            if kwargs["pred_answer"] != None:
                pred_answer = [int(kwargs["pred_answer"].squeeze().detach().numpy())]
                pred_answer_decoded = self.dataset.answer_tokenizer.decode(text=pred_answer)
                values["pred_answer"] = pred_answer_decoded

            dialog = ["{} : {}".format(key, value) for key, value in values.items()]
            dialog.append("-" * 70)
            dialog = " \n ".join(dialog)
            self.metric.append(dialog)

            id = self.get_id_image(kwargs["img_idx"])
            # if id is not None:
            url = "https://drive.google.com/uc?export=view&id={}".format(id)
            values["link"] = "<img src={}>".format(url)
            values["closest_question"] = kwargs["closest_question"]
            self.generated_dialog.append(values)

    def get_id_image(self, id_image):

        try:
            if self.dataset.__class__ == CLEVR_Dataset:
                mode = 0 if self.mode == "train" else 1
                id_drive = self.list_image_ids.loc[id_image]["id_google"].iloc[mode]
            else:
                id_drive = self.list_image_ids.loc[id_image]["id_google"]

        except :
            id_drive = None
        finally:
            return id_drive

    def post_treatment_(self):
        num_last_episodes = min(50, len(self.generated_dialog))
        to_html = lambda x: "".join(["<li>{} : {}</li>".format(key, value) for key, value in x.items()])
        html_str = ["<tr><ul>" + to_html(x) + "</ul></tr>" for x in self.generated_dialog[-num_last_episodes:]]
        html_str = "".join(html_str)
        html_str = "<table>" + html_str + "</table>"
        f = open(self.out_html_file, "x")
        f.write(html_str)
        f.close()


class PPLMetric(Metric):
    """
    Compute the ppl of the learning policy on the ref questions.
    https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
    """

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "ppl", "scalar", env_mode, trunc, sampling)
        self.agent = agent

    def get_stats(self, serie):
        return [serie.median(), serie.std(), serie.size]

    def fill_(self, **kwargs):
        if kwargs["done"]:
            with torch.no_grad():
                state = kwargs["state"]
                sos = torch.tensor([self.dataset.question_tokenizer.vocab["<SOS>"]])
                ref_question = kwargs["ref_question"][kwargs["ref_question"] != 0]
                # getting the probs for the complete policy
                ref_question = torch.cat((sos, ref_question), dim=-1).unsqueeze(dim=0)
                ht, ct = self.agent.init_hidden(state)
                for i, action in enumerate(ref_question[:, 1:].view(-1)):
                    forced_state = state.__class__(ref_question[:, :i + 1], state.img, state.answer)

                    real_action, log_probs, _, _, dist, _, _, _, ht, ct = self.agent.act(
                        state=forced_state,
                        mode="forced",
                        truncation=True,
                        forced=action, ht=ht, ct=ct)
                    self.measure.append(log_probs)

    def compute_(self, **kwargs):
        ppl = torch.exp(-torch.stack(self.measure).sum() / len(self.measure)).cpu().numpy().item()
        self.metric.append(ppl)


class PPLDialogfromLM(Metric):
    '''Computes the PPL of the Language Model over the generated dialog'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "ppl_dialog_lm", "scalar", env_mode, trunc, sampling)
        self.min_data = agent.env.min_data
        self.device = agent.device
        self.get_lm_model(agent)

    def get_lm_model(self, agent):
        if self.dataset.__class__ == CLEVR_Dataset:
            lm_model = torch.load("output/lm_model/model.pt", map_location=torch.device('cpu'))
        else:
            if self.min_data:
                lm_model = torch.load("output/vqa_lm_model_smallvocab/model.pt", map_location=torch.device('cpu'))
            else:
                lm_model = torch.load("output/vqa_lm_model/model.pt", map_location=torch.device('cpu'))
        lm_model.eval()
        self.pretrained_lm = ClevrLanguageModel(pretrained_lm=lm_model, dataset=self.dataset,
                                                tokenizer=self.dataset.question_tokenizer, device=agent.device)

    def fill_(self, **kwargs):
        with torch.no_grad():
            log_probas_lm, _, _ = self.pretrained_lm.forward(kwargs["state"].text.to(self.device), temperature=1)
            log_probas_lm = log_probas_lm.cpu()
            self.measure.append(log_probas_lm[:, kwargs["action"]])

    def compute_(self, **kwargs):
        if len(self.measure) > 0:
            ppl = torch.exp(-torch.stack(self.measure).sum() / len(self.measure)).cpu().numpy().item()
            self.metric.append(ppl)


class LanguageScore(Metric):
    '''Compute the perplexity of a pretrained language model (GPT) on the generated dialog.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "language_score", "scalar", env_mode, trunc, sampling)
        self.lm_model = AutoModelWithLMHead.from_pretrained("cache/gpt-2")
        self.tokenizer = AutoTokenizer.from_pretrained("cache/gpt-2")

    def fill_(self, **kwargs):
        if kwargs["state"].text.shape[-1] > 1:
            state_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text[:, 1:].cpu().numpy()[0])
            inputs = self.tokenizer(state_decoded, return_tensors="pt")
            if inputs["input_ids"].shape[-1] >= 1:
                logits = self.lm_model(inputs["input_ids"])[0]
                scores = F.log_softmax(logits, dim=-1)  # (B, S, vocab size)
                action_decoded = self.dataset.question_tokenizer.decode(kwargs["action"].cpu().numpy())
                action_id = self.tokenizer(action_decoded)
                # TODO: add the case if len(action_id["input_ids"]) == 0 (pad TOKEN).
                if len(action_id["input_ids"]) == 1:
                    log_prob = scores[:, -1, action_id["input_ids"][0]]
                    self.measure.append(log_prob.squeeze())
                elif len(action_id["input_ids"]) == 2:
                    log_prob_1 = scores[:, -1, action_id["input_ids"][0]]
                    self.measure.append(log_prob_1.squeeze())
                    inputs_2 = torch.cat([inputs["input_ids"], torch.tensor(action_id["input_ids"][0]).view(1, 1)],
                                         dim=-1)
                    outputs = self.lm_model(input_ids=inputs_2)
                    scores_2 = F.log_softmax(outputs[0], dim=-1)  # (B, S, vocab size)
                    log_prob_2 = scores_2[:, -1, action_id["input_ids"][1]]
                    self.measure.append(log_prob_2.squeeze())

    def compute_(self, **kwargs):
        if len(self.measure) > 0:
            ppl = torch.exp(-torch.stack(self.measure).sum() / len(self.measure)).cpu().numpy().item()
            self.metric.append(ppl)


class Return(Metric):
    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "return", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        self.measure.append(kwargs["reward"])

    def compute_(self, **kwargs):
        ep_return = np.sum(self.measure)
        self.metric.append(ep_return)


class BleuMetric(Metric):
    '''Compute the bleu score over the ref questions and the generated dialog.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "bleu", "scalar", env_mode, trunc, sampling)
        if "bleu" in agent.env.reward_type:
            self.function = agent.env.reward_func
        else:
            self.function = rewards["bleu_sf2"]()

    def fill_(self, **kwargs):
        if kwargs["done"]:
            question_decoded = self.dataset.question_tokenizer.decode(kwargs["new_state"].text.numpy()[0],
                                                                      ignored=["<SOS>"],
                                                                      stop_at_end=True)
            ref_questions = kwargs["ref_questions_decoded"]
            score, _, _ = self.function.get(ep_questions_decoded=ref_questions, question=question_decoded, done=True)
            self.measure.append(score)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class SelfBleuImageMetric(Metric):
    '''Compute the self bleu score on generated sentences for one image'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "selfbleu", "scalar", env_mode, trunc, sampling)
        if "bleu" in agent.env.reward_type:
            self.function = agent.env.reward_func
        else:
            self.function = rewards["bleu_sf2"]()
        self.questions = []
        self.out_questions_csv_file = os.path.join(self.out_path, "metrics", self.name + "_questions.csv")

    def fill_(self, **kwargs):
        pass

    def compute_(self, **kwargs):
        if self.sampling != "greedy":
            question_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text.numpy()[0],
                                                                      ignored=["<SOS>"],
                                                                      stop_at_end=True)
            self.questions.append(question_decoded)
            if (kwargs["idx_diversity"] == kwargs["num_diversity"] - 1) and kwargs["num_diversity"] > 1:
                scores = []
                for i, question in enumerate(self.questions):
                    ref_questions = np.delete(self.questions, i)
                    score, _, _ = self.function.get(ep_questions_decoded=ref_questions, question=question, done=True)
                    scores.append(score)
                self.metric.append(np.mean(scores))
                self.questions = []


class HistogramOracle(Metric):
    """Compute the Histogram of Correct Answers."""

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "histogram_answers", "text", env_mode, trunc, sampling)
        self.metric_history = {}
        self.answer_history = {}
        self.out_png_file = os.path.join(self.out_path, "metrics", self.name + ".png")

    def fill_history(self, history, ref_answer_decoded, reward):
        if ref_answer_decoded in history.keys():
            history[ref_answer_decoded] += reward
        else:
            history[ref_answer_decoded] = reward
        return history

    def fill_(self, **kwargs):
        if self.reward_type == "vilbert" or self.reward_type == "vqa":
            if kwargs["done"]:
                ref_answer_decoded = self.dataset.answer_tokenizer.decode(kwargs["ref_answer"].view(1).numpy())
                if kwargs["reward"] == 1.:
                    self.metric_history = self.fill_history(self.metric_history, ref_answer_decoded, kwargs["reward"])
                    self.answer_history = self.fill_history(self.answer_history, ref_answer_decoded, 1)
                else:
                    self.answer_history = self.fill_history(self.answer_history, ref_answer_decoded, 1)

    def compute_(self, **kwargs):
        pass

    def write(self, **kwargs):
        pass

    def get_top_k_values(self, k=25):
        if len(self.metric_history) > k:
            k_keys_sorted_by_values = heapq.nlargest(k, self.metric_history, key=self.metric_history.get)
            top_k_dict = {k: v for k, v in self.metric_history.items() if k in k_keys_sorted_by_values}
        else:
            top_k_dict = self.metric_history
        return top_k_dict

    def post_treatment(self, num_episodes):
        if self.reward_type == "vilbert" or self.reward_type == "vqa":
            self.post_treatment_()
            metric_history_sorted = dict(sorted(self.metric_history.items(), key=operator.itemgetter(1), reverse=True))
            answer_history_sorted = {k: self.answer_history[k] for k in list(metric_history_sorted.keys())}
            df = pd.DataFrame.from_dict([metric_history_sorted, answer_history_sorted])
            df = df.transpose()
            df.to_csv(self.out_csv_file, index=True, header=["reward = 1", "answer freq"])

    def post_treatment_(self):
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(30, 15))
        top_k_metric_history = self.get_top_k_values()
        top_k_answer_history = {k: self.answer_history[k] for k in list(top_k_metric_history.keys())}
        ax1.bar(list(top_k_metric_history.keys()), top_k_metric_history.values())
        ax2.bar(list(top_k_metric_history.keys()), top_k_answer_history.values())
        ax1.tick_params(labelsize=18)
        ax2.tick_params(labelsize=18)
        plt.savefig(self.out_png_file)


class LvNormMetric(Metric):
    '''Compute the levenshtein over the ref questions and the generated dialog.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "lv_norm", "scalar", env_mode, trunc, sampling)
        self.function = rewards["lv_norm"]()

    def fill_(self, **kwargs):
        if kwargs["done"]:
            question_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text.numpy()[0],
                                                                      ignored=["<SOS>"],
                                                                      stop_at_end=True)
            ref_questions = kwargs["ref_questions_decoded"]
            score, _, _ = self.function.get(ep_questions_decoded=ref_questions, question=question_decoded,
                                            step_idx=kwargs["timestep"], done=True)
            self.measure.append(score)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


# ------------------------ DIVERSITY METRICS -------------------------------------------------------------------------------------------------------------------

class RefQuestionsMetric(Metric):
    '''
    Compute the ratio of Unique closest questions on all the set of questions generated for the same image.
    '''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "ratio_closest_questions", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        if kwargs["done"]:
            self.measure.append(kwargs["closest_question"])

    def compute_(self, **kwargs):
        unique_ratio = len(list(set(self.measure))) / len(self.measure)
        self.measure.append(unique_ratio)

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

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "ttr_question", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        if kwargs["done"]:
            self.measure = kwargs["new_state"].text.numpy()[0]

    def compute_(self, **kwargs):
        diversity_metric = len(set(list(self.measure))) / len(self.measure)
        self.metric.append(diversity_metric)


class TrueWordRankOriginLM(Metric):
    """
    Compute the rank of the true word in the original lm logits
    """

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "true_word_rank", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        true_action = kwargs["ref_question"].view(-1)[kwargs["timestep"]].cpu().numpy().item()
        if kwargs["origin_log_probs_lm"] is not None and true_action != 0:
            true_lm_action = self.language_model.dataset_to_lm_trad[true_action]
            sorted, indices = torch.sort(kwargs["origin_log_probs_lm"][:, -1, :], descending=True)
            rank = int(torch.nonzero(indices.squeeze().cpu() == true_lm_action).squeeze().numpy())
            self.measure.append(rank)

    def compute_(self, **kwargs):
        self.metric.extend(self.measure)


class TrueWordRankLM(Metric):
    """
    Compute the rank of the target word in the lm logits
    """

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "true_word_rank", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        true_action = kwargs["ref_question"].view(-1)[kwargs["timestep"]].cpu().numpy().item()
        if kwargs["log_probas_lm"] is not None and true_action != 0:
            sorted, indices = torch.sort(kwargs["log_probas_lm"].squeeze(), descending=True)
            rank = int(torch.nonzero(indices.squeeze().cpu() == true_action).squeeze().numpy())
            self.measure.append(rank)

    def compute_(self, **kwargs):
        self.metric.extend(self.measure)


class ActionRankLM(Metric):
    """
    Compute the rank of the action taken in the original lm logits
    """

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "true_word_rank", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        true_action = kwargs["ref_question"].view(-1)[kwargs["timestep"]].cpu().numpy().item()
        if kwargs["origin_log_probs_lm"] is not None and true_action != 0:
            true_action = kwargs["action"].cpu().numpy().item()
            true_lm_action = self.language_model.dataset_to_lm_trad[true_action]
            sorted, indices = torch.sort(kwargs["origin_log_probs_lm"][:, -1, :], descending=True)
            rank = int(torch.nonzero(indices.squeeze().cpu() == true_lm_action).squeeze().numpy())
            self.measure.append(rank)

    def compute_(self, **kwargs):
        self.metric.extend(self.measure)


class TrueWordProbLM(Metric):
    """
    Compute the probability of the true word in the original lm logits
    """

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "true_word_prob", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        true_action = kwargs["ref_question"].view(-1)[kwargs["timestep"]].cpu().numpy().item()
        if kwargs["origin_log_probs_lm"] is not None and true_action != 0:
            true_action = kwargs["ref_question"].view(-1)[kwargs["timestep"]].cpu().numpy().item()
            true_lm_action = self.language_model.dataset_to_lm_trad[true_action]
            prob = kwargs["origin_log_probs_lm"][:, -1, true_lm_action].exp().cpu().numpy()[0]
            self.measure.append(prob)

    def compute_(self, **kwargs):
        self.metric.extend(self.measure)


class UniqueWordsMetric(Metric):
    '''Compute the ratio of Unique Words for the set of questions generated for each image. Allows to measure vocabulary diversity.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "ttr", "scalar", env_mode, trunc, sampling)
        self.measure_history = []
        self.threshold = 10

    def fill_(self, **kwargs):
        if kwargs["done"]:
            self.measure_history.append(list(kwargs["new_state"].text.squeeze().cpu().numpy()[1:]))

    def compute_(self, **kwargs):
        if self.idx_compute > self.threshold and "sampling" in self.id:
            arr = [item for sublist in self.measure_history[-self.threshold:] for item in sublist]
            unique_tokens = np.unique(arr)
            diversity_metric = len(unique_tokens) / len(arr) if len(arr) > 0 else 0
            self.metric.append(diversity_metric)


# --------------------------------------- OLD METRICS ----------------------------------------------------------------------------------------------------

class PolicyMetric(Metric):
    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "policy", "text", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        # compute top_k_words from the Policy:
        with torch.no_grad():
            state_decoded = self.dataset.question_tokenizer.decode(tex=kwargs["state"].text.numpy()[0],
                                                                   ignored=[])
            top_k_weights, top_k_indices = torch.topk(kwargs["dist"].probs, 5, sorted=True)
            top_words_decoded = self.question_tokenizer.decode(tex=top_k_indices.cpu().numpy()[0])
            # get top_words from the language model:
            seq_len = kwargs["state"].text.size(1)
            log_probas, _ = self.language_model.forward(kwargs["state"].text.to(self.agent.device))
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
        logger.info('---------------------Policy Top Words------------------------------')
        logger.info('\n'.join(self.metric))
        logger.info('--------------------------------------------------------------------')


class ValidActionsMetric(Metric):
    """Look at the mismatch ref question / valid action space per episode."""

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "valid_actions_episode", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            ref_question = kwargs["ref_question"][kwargs["ref_question"] != 0].cpu()
            if len(ref_question) > self.idx_word:
                if ref_question[self.idx_word] not in kwargs["valid_actions"].cpu():
                    self.measure.append(0)
                else:
                    self.measure.append(1)

    def compute_(self, **kwargs):
        self.metric.append(np.sum(self.measure) / len(self.measure))


class LMVAMetric(Metric):
    '''Monitor the mismatch between the valid actions space and the ref questions.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "lm_valid_actions", "scalar", env_mode, trunc, sampling)
        self.counter = 0

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            ref_question = kwargs["ref_question"][kwargs["ref_question"] != 0].cpu()
            if len(ref_question) > self.idx_word:
                if ref_question[self.idx_word] not in kwargs["valid_actions"].cpu():
                    self.counter += 1

    def compute_(self, **kwargs):
        self.metric = [self.counter]


class PoliciesRatioMetric(Metric):
    '''to monitor the discrepancy between the truncated policy (used for action selection) and the learned policy'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "policies_discrepancy", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        ratios = np.exp(
            kwargs["log_probs"].detach().cpu().numpy() - kwargs["log_probs_truncated"].detach().cpu().numpy())
        self.measure.append(ratios)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class LMPolicyProbsRatio(Metric):
    '''to monitor the difference between the proba given by the lm for the words choosen and the probas given by the policy.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "lm_policy_probs_ratio", "scalar", env_mode, trunc, sampling)

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
    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "action_probs", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        self.measure.append(kwargs["log_probs"])

    def compute_(self, **kwargs):
        ep_log_probs = torch.stack(self.measure).clone().detach()
        self.ep_probs = np.round(np.exp(ep_log_probs.cpu().squeeze().numpy()), decimals=5)
        self.metric.append(np.mean(self.ep_probs))

    def log(self, **kwargs):
        logger.info('episode action probs: {}'.format(self.ep_probs))


class ActionProbsTruncated(Metric):
    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "action_probs_truncated", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        action_decoded = self.dataset.question_tokenizer.decode(kwargs["action"].cpu().numpy())
        self.measure.append([kwargs["log_probs_truncated"], action_decoded])

    def compute_(self, **kwargs):
        self.ep_probs_truncated = {}
        logprobs = [i[0] for i in self.measure]
        ep_log_probs_truncated = torch.stack(logprobs).clone().detach()
        self.ep_probs_truncated["actions"] = [i[1] for i in self.measure]
        self.ep_probs_truncated["probs"] = np.round(np.exp(ep_log_probs_truncated.cpu().view(-1).numpy()), decimals=5)
        self.metric.append(np.mean(self.ep_probs_truncated["probs"]))

    def log(self, **kwargs):
        log_info = ["{}/{:.3f}".format(word, weight, number=3) for word, weight in
                    zip(self.ep_probs_truncated["actions"], self.ep_probs_truncated["probs"])]
        logger.info('episode action probs truncated: {}'.format(log_info))


class LMActionProbs(Metric):
    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "action_probs_lm", "scalar", env_mode, trunc, sampling)

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
        logger.info('episode action probs from the LANGUAGE MODEL: {}'.format(self.ep_lm_probs))


class OracleMetric(Metric):
    '''Compute the oracle score over the ref answer and the generated dialog.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "oracle", "scalar", env_mode, trunc, sampling)
        if agent.env.reward_type in ["vilbert", "vqa"]:
            self.function = agent.env.reward_func
        else:
            if self.dataset.__class__ != CLEVR_Dataset:
                self.function = rewards["vilbert"](path="output/vilbert_vqav2/model.bin",
                                                   vocab="output/vilbert_vqav2/bert_base_6layer_6conect.json",
                                                   env=agent.env)
            else:
                self.function = rewards["vqa"](path="output/vqa_model_film/model.pt",
                                               vocab="data/closure_vocab.json",
                                               dataset=agent.env.dataset)

    def fill_(self, **kwargs):
        if kwargs["done"]:
            if self.dataset.__class__ != CLEVR_Dataset:

                question_decoded = self.dataset.question_tokenizer.decode(kwargs["new_state"].text.numpy()[0],
                                                                          ignored=["<SOS>"],
                                                                          stop_at_end=True)
                ref_questions = kwargs["ref_questions_decoded"][0]
                score, _, _ = self.function.get(ep_questions_decoded=ref_questions, question=question_decoded,
                                                step_idx=None,
                                                done=True)
            else:
                question_decoded = self.dataset.question_tokenizer.decode(kwargs["new_state"].text.numpy()[0],
                                                                          ignored=["<SOS>"],
                                                                          stop_at_end=True)

                ref_questions = kwargs["ref_questions_decoded"][0]
                score, _, _ = self.function.get(ep_questions_decoded=ref_questions, question=question_decoded,
                                                step_idx=None,
                                                done=True, state=kwargs["new_state"],
                                                real_answer=kwargs["state"].answer.view(-1))
            self.measure.append(score)

    def compute_(self, **kwargs):
        if len(self.measure) > 0:
            self.metric.append(np.mean(self.measure))


class OracleRecallMetric(OracleMetric):
    '''Compute the oracle score over the ref answer and the generated dialog.'''

    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "oracle_recall", "scalar", env_mode, trunc, sampling)
        if agent.env.reward_type in ["vilbert_recall", "vqa_recall"]:
            self.function = agent.env.reward_func
        else:
            if self.dataset.__class__ != CLEVR_Dataset:
                self.function = rewards["vilbert_recall"](path="output/vilbert_vqav2/model.bin",
                                                          vocab="output/vilbert_vqav2/bert_base_6layer_6conect.json",
                                                          env=agent.env)
            else:
                self.function = rewards["vqa_recall"](path="output/vqa_model_film/model.pt",
                                                      vocab="data/closure_vocab.json",
                                                      dataset=agent.env.dataset)


class CiderMetric(Metric):
    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "cider", "scalar", env_mode, trunc, sampling)
        self.score_function = Cider()
        self.tokenizer = PTBTokenizer()
        self.candidates = []
        self.refs = []

    def fill_(self, **kwargs):
        pass

    def compute_(self, **kwargs):
        question_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text.numpy()[0],
                                                                  ignored=["<SOS>"], stop_at_end=True)
        ref_questions = kwargs["ref_questions_decoded"][0]
        self.candidates.append(question_decoded)
        self.refs.append([ref_questions])

    def post_treatment_(self):
        refs = {idx: list(map(_strip, ref)) for (idx, ref) in enumerate(self.refs)}
        hyps = {idx: [lines.strip()] for (idx, lines) in enumerate(self.candidates)}
        score, scores = self.score_function.compute_score(refs, hyps)
        self.metric_history.extend(scores)


class MeteorMetric(Metric):
    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "meteor", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        pass

    def compute_(self, **kwargs):
        question_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text.numpy()[0],
                                                                  ignored=["<SOS>"], stop_at_end=True)
        ref_questions = kwargs["ref_questions_decoded"]
        score = meteor_score(references=ref_questions, hypothesis=question_decoded)
        self.metric.append(score)


class KurtosisMetric(Metric):
    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "kurtosis", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        dist = pd.Series(kwargs["dist"].probs.squeeze().cpu().numpy())
        self.measure.append(dist.kurtosis())

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class PeakinessMetric(Metric):
    def __init__(self, agent, train_test, env_mode, trunc, sampling):
        Metric.__init__(self, agent, train_test, "peakiness", "scalar", env_mode, trunc, sampling)

    def fill_(self, **kwargs):
        sorted, indices = torch.sort(kwargs["dist"].probs.cpu(), descending=True)
        sum_10 = sorted[:, :10].sum().item()
        self.measure.append(sum_10)

    def compute_(self, **kwargs):
        self.metric.extend(self.measure)


metrics = {"return": Return, "valid_actions": VAMetric, "size_valid_actions": SizeVAMetric,
           "dialog": DialogMetric, "dialogimage": DialogImageMetric,
           "ppl": PPLMetric, "ppl_dialog_lm": PPLDialogfromLM, "bleu": BleuMetric,
           "ttr_question": TTRQuestionMetric, "sum_probs": SumProbsOverTruncated, "true_word_rank": TrueWordRankLM,
           "true_word_prob": TrueWordProbLM, "lv_norm": LvNormMetric, "ttr": UniqueWordsMetric,
           "selfbleu": SelfBleuImageMetric, "language_score": LanguageScore,
           "action_probs_truncated": ActionProbsTruncated,
           "lm_valid_actions": LMVAMetric, "valid_actions_episode": ValidActionsMetric,
           "histogram_answers": HistogramOracle, "oracle": OracleMetric, "cider": CiderMetric, "meteor": MeteorMetric,
           "kurtosis": KurtosisMetric, "oracle_recall": OracleRecallMetric, "peakiness": PeakinessMetric}
metrics_to_tensorboard = ["return", "size_valid_actions", "sum_probs_truncated", "lm_valid_actions", "ttr",
                          "action_probs_truncated", "valid_actions_episode", "ppl_dialog_lm", "ttr_question"]
