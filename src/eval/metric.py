import logging
import os
import pickle

import h5py
import numpy as np
import pandas as pd
import torch
from google.auth.transport.requests import Request
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from nltk.translate.bleu_score import sentence_bleu
from torch.nn.utils.rnn import pad_sequence

# If modifying these scopes, delete the file token.pickle.

SCOPES = ['https://www.googleapis.com/auth/drive.metadata.readonly']


class Metric:
    def __init__(self, agent, train_test, key, type):
        self.measure = []
        self.metric = []
        self.metric_history = []
        self.idx_step = 0
        self.idx_word = 0
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
        # self.dict_metric, self.dict_stats = {}, {}  # for csv writing.
        self.out_csv_file = os.path.join(self.out_path, "metrics", self.train_test + '_' + self.key + ".csv")
        self.stats_path = os.path.join(self.out_path, "stats", self.train_test + "_" + self.key + '_stats.csv')
        self.stats = None
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

    def reset(self):
        self.idx_word = 0

    def reinit_train_test(self, train_test):
        self.train_test = train_test

    def write(self, **kwargs):
        if self.to_tensorboard:
            if self.type == "scalar":
                self.writer.add_scalar(self.train_test + "_" + self.key, np.mean(self.metric), self.idx_write)
            elif self.type == "text":
                self.writer.add_text(self.train_test + "_" + self.key, '  \n'.join(self.metric[-1:]), self.idx_write)
        self.idx_write += 1
        self.metric_history.extend(self.metric)
        self.metric = []

    def log(self, **kwargs):
        pass

    def post_treatment(self):
        serie = pd.Series(self.metric_history)
        serie.to_csv(self.out_csv_file, index=False)
        if self.type == "scalar":
            self.stats = [serie.mean(), serie.std(), serie.size]
            pd.Series(self.stats).to_csv(self.stats_path, index=False)


# ----------------------------------  TRAIN METRICS -------------------------------------------------------------------------------------

class VAMetric(Metric):
    '''Display the valid action space in the training log.'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "valid_actions", "text")

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
        self.metric.extend(self.measure[0])

    def log(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            logging.info('---------------------Valid action space------------------------------')
            logging.info('\n'.join(self.metric))
            logging.info('---------------------------------------------------------------------')

    def write(self):
        pass


class SizeVAMetric(Metric):
    '''Compute the average size of the truncated action space during training for truncation functions proba_thr & sample_va'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "size_valid_actions", "scalar")
        self.counter = 0

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            self.measure.append(kwargs["valid_actions"].size(1))

    def compute_(self, **kwargs):
        self.metric.extend(self.measure)


class SumProbsOverTruncated(Metric):
    '''Compute the sum of the probabilities the action space given by the language model.'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "sum_probs_truncated", "scalar")

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

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "dialog", "text")
        self.out_dialog_file = os.path.join(self.out_path, self.train_test + '_' + self.key + '.txt')
        self.h5_dialog_file = os.path.join(self.out_path, self.train_test + '_' + self.key + '.h5')
        self.generated_dialog = {}

    def fill_(self, **kwargs):
        pass

    def reinit_train_test(self, train_test):
        self.train_test = train_test
        self.out_dialog_file = os.path.join(self.out_path, self.train_test + '_' + self.key + '.txt')

    def compute_(self, **kwargs):
        with torch.no_grad():
            state_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text[:, 1:].numpy()[0],
                                                                   ignored=[])
            if self.reward_type == 'vqa':
                pred_answer_decoded = self.dataset.question_tokenizer.decode(kwargs["pred_answer"].numpy(),
                                                                             decode_answers=True)
                ref_answer_decoded = self.dataset.question_tokenizer.decode(
                    text=[kwargs["ref_answer"].numpy().item()],
                    decode_answers=True)
                ref_question_decoded = kwargs["ref_questions_decoded"]
                string = ' IMG {} - question index {}:'.format(kwargs["img_idx"],
                                                               kwargs["question_idx"]) \
                         + '\n' + 'DIALOG:' + state_decoded + '\n' + 'VQA ANSWER:' + pred_answer_decoded + '\n' + 'TRUE ANSWER:' + ref_answer_decoded + '\n' + 'REF QUESTION:' + \
                         ref_question_decoded[0] + '\n' + '-' * 40
            else:
                closest_question_decoded = kwargs["closest_question"]
                string = 'IMG {}:'.format(kwargs[
                                              "img_idx"]) + state_decoded + '\n' + 'CLOSEST QUESTION:' + closest_question_decoded + '\n' + '-' * 40
            self.metric.append(string)
            # write dialog in a .txt file:
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


class DialogImageMetric(Metric):
    '''Display the Dialog on a html format at test time.'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "dialog", "text")
        self.out_dialog_file = os.path.join(self.out_path, self.train_test + '_' + self.key + '.html')
        self.h5_dialog_file = os.path.join(self.out_path, self.train_test + '_' + self.key + '.h5')
        self.generated_dialog = {}
        self.drive_service = self.get_google_service()

    def fill_(self, **kwargs):
        pass

    def reinit_train_test(self, train_test):
        self.train_test = train_test
        self.out_dialog_file = os.path.join(self.out_path, self.train_test + '_' + self.key + '.html')

    def compute_(self, **kwargs):
        with torch.no_grad():
            if not self.train_test + '_' + self.key in self.generated_dialog.keys():
                self.generated_dialog[self.train_test + '_' + self.key] = [kwargs["state"].text.squeeze().cpu()]
            else:
                self.generated_dialog[self.train_test + '_' + self.key].append(
                    kwargs["state"].text.cpu().view(-1))
            state_decoded = self.dataset.question_tokenizer.decode(tex=kwargs["state"].text[:, 1:].numpy()[0],
                                                                   ignored=[])
            if self.env.reward_type == 'vqa':
                pred_answer_decoded = self.dataset.question_tokenizer.decode(tex=kwargs["pred_answer"].numpy(),
                                                                             decode_answers=True)
                ref_answer_decoded = self.dataset.question_tokenizer.decode(
                    tex=[kwargs["ref_answer"].numpy().item()],
                    decode_answers=True)
                ref_question_decoded = kwargs["ref_questions_decoded"][0]

                values = [kwargs["img_idx"], kwargs["question_idx"], state_decoded, pred_answer_decoded,
                          ref_answer_decoded, ref_question_decoded]
            else:
                values = [kwargs["img_idx"], state_decoded, kwargs["closest_question"]]
            string = '<table><tr>'
            img_name = "CLEVR_{}_{:06d}.png".format(self.env.clevr_mode, kwargs["img_idx"])
            id = self.get_id_image(img_name)
            url = "https://drive.google.com/uc?export=view&id={}".format(id)
            values.append("<img src={}>".format(url))

            string += "<td><ul><li>" + "</li><li>".join(list(map(str, values))) + "</li></ul></td></tr></table>"

            self.metric.append(string)
            # write dialog in a .html file:
            with open(self.out_dialog_file, 'a') as f:
                f.write(string + '\n')
            pass

    def get_id_image(self, name):
        page_token = None
        id = "unknown"
        while True:
            try:
                response = self.drive_service.files().list(q="name = '{}'".format(name),
                                                           spaces='drive',
                                                           fields='nextPageToken, files(id, name)',
                                                           pageToken=page_token).execute()
                file = response["files"][0]
                id = file.get('id')

            except Exception as e:
                print(e)
            break
        return id

    def get_google_service(self):
        creds = None
        # The file token.pickle stores the user's access and refresh tokens, and is
        # created automatically when the authorization flow completes for the first
        # time.
        if os.path.exists('token.pickle'):
            with open('token.pickle', 'rb') as token:
                creds = pickle.load(token)
        # If there are no (valid) credentials available, let the user log in.
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                creds.refresh(Request())
            else:
                flow = InstalledAppFlow.from_client_secrets_file(
                    'credentials.json', SCOPES)
                creds = flow.run_local_server(port=0)
            # Save the credentials for the next run
            with open('token.pickle', 'wb') as token:
                pickle.dump(creds, token)

        service = build('drive', 'v3', credentials=creds)
        return service


class PPLMetric(Metric):
    """
    Compute the ppl of the learning policy on the ref questions.
    https://towardsdatascience.com/perplexity-in-language-models-87a196019a94
    """

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "ppl", "scalar")

    def fill_(self, **kwargs):
        if kwargs["done"]:
            with torch.no_grad():
                input_ids = kwargs["ref_question"].view(1, -1)
                state = kwargs["state"]
                # getting the probs for the complete policy
                policy_dist, _, _ = self.policy(state.text, state.img, state.answer, logits_lm=kwargs["logits_lm"],
                                                alpha=kwargs["alpha"])
                log_prob_actions = torch.gather(policy_dist.probs.detach().cpu(), -1, input_ids.cpu())
                self.measure.extend(log_prob_actions.view(-1))

    def compute_(self, **kwargs):
        ppl = torch.exp(-torch.stack(self.measure).sum() / len(self.measure)).cpu().numpy().item()
        self.metric.append(ppl)


class PPLDialogfromLM(Metric):
    '''Computes the PPL of the Language Model over the generated dialog'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "ppl_dialog_lm", "scalar")

    def fill_(self, **kwargs):
        if kwargs["log_probas_lm"] is not None:
            self.measure.append(kwargs["log_probas_lm"][:, kwargs["action"]])

    def compute_(self, **kwargs):
        if len(self.measure) > 0:
            ppl = torch.exp(-torch.stack(self.measure).sum() / len(self.measure)).cpu().numpy().item()
            self.metric.append(ppl)


class Return(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "return", "scalar")

    def fill_(self, **kwargs):
        self.measure.append(kwargs["reward"])

    def compute_(self, **kwargs):
        ep_return = np.sum(self.measure)
        self.metric.append(ep_return)


class BleuMetric(Metric):
    '''Compute the bleu score over the ref questions and the generated dialog.'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "bleu", "scalar")

    def fill_(self, **kwargs):
        if kwargs["done"]:
            question_decoded = self.dataset.question_tokenizer.decode(kwargs["state"].text.numpy()[0],
                                                                      ignored=["<SOS>"],
                                                                      stop_at_end=True)
            ref_questions = kwargs["ref_questions_decoded"]
            ref_questions = [q.split() for q in ref_questions]
            question_tokens = question_decoded.split()
            score = sentence_bleu(ref_questions, question_tokens)
            self.measure.append(score)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


# ------------------------ DIVERSITY METRICS -------------------------------------------------------------------------------------------------------------------

class RefQuestionsMetric(Metric):
    '''
    Compute the ratio of Unique closest questions on all the set of questions generated for the same image.
    '''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "ratio_closest_questions", "scalar")

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

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "ttr_question", "scalar")

    def fill_(self, **kwargs):
        if kwargs["done"]:
            self.measure = kwargs["new_state"].text.numpy()[0]

    def compute_(self, **kwargs):
        diversity_metric = len(set(list(self.measure))) / len(self.measure)
        self.metric.append(diversity_metric)


class TrueWordRankLM(Metric):
    """
    Compute the rank of the true word in the original lm logits
    """

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "true_word_rank", "scalar")

    def fill_(self, **kwargs):
        if kwargs["origin_log_probs_lm"] is not None:
            true_action = kwargs["action"].cpu().numpy().item()
            true_lm_action = self.language_model.dataset_to_lm_trad[true_action]
            sorted, indices = torch.sort(kwargs["origin_log_probs_lm"][:, -1, :], descending=True)
            rank = int((indices.squeeze().cpu() == true_lm_action).nonzero().squeeze().numpy())
            self.measure.append(rank)

    def compute_(self, **kwargs):
        self.metric.extend(self.measure)


class TrueWordProbLM(Metric):
    """
    Compute the probability of the true word in the original lm logits
    """

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "true_word_prob", "scalar")

    def fill_(self, **kwargs):
        if kwargs["origin_log_probs_lm"] is not None:
            true_action = kwargs["action"].cpu().numpy().item()
            true_lm_action = self.language_model.dataset_to_lm_trad[true_action]
            prob = kwargs["origin_log_probs_lm"][:, -1, true_lm_action].exp().cpu().numpy()[0]
            self.measure.append(prob)

    def compute_(self, **kwargs):
        self.metric.extend(self.measure)


class UniqueWordsMetric(Metric):
    '''Compute the ratio of Unique Words for the set of questions generated for each image. Allows to measure vocabulary diversity.'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "unique_words", "scalar")

    def fill_(self, **kwargs):
        if kwargs["done"]:
            self.measure.append(list(kwargs["new_state"].text.numpy()[0]))

    def compute_(self, **kwargs):
        if "sampling" in self.train_test:
            if len(self.measure) == kwargs["ref_question"].size(0):
                arr = np.array(self.measure).flatten()
                unique_tokens = np.unique(arr)
                diversity_metric = len(unique_tokens) / len(arr)
                self.metric.append(diversity_metric)

    def compute(self, **kwargs):
        self.compute_(**kwargs)
        self.idx_word = 0
        self.idx_step = 0

    def write(self):
        pass


# --------------------------------------- OLD METRICS ----------------------------------------------------------------------------------------------------

class PolicyMetric(Metric):
    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "policy", "text")

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
        logging.info('---------------------Policy Top Words------------------------------')
        logging.info('\n'.join(self.metric))
        logging.info('--------------------------------------------------------------------')


class LMVAMetric(Metric):
    '''Monitor the mismatch between the valid actions space and the ref questions.'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "lm_valid_actions", "scalar")
        self.counter = 0

    def fill_(self, **kwargs):
        if kwargs["valid_actions"] is not None:
            closest_question = self.question_tokenizer.decode(kwargs["closest_question"].split())
            if len(closest_question) > self.idx_word:
                if closest_question[self.idx_word] not in kwargs["valid_actions"]:
                    self.counter += 1
                    logging.info("+VA")

    def compute_(self, **kwargs):
        self.metric = [self.counter]


class PoliciesRatioMetric(Metric):
    '''to monitor the discrepancy between the truncated policy (used for action selection) and the learned policy'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "policies_discrepancy", "scalar")

    def fill_(self, **kwargs):
        ratios = np.exp(
            kwargs["log_probs"].detach().cpu().numpy() - kwargs["log_probs_truncated"].detach().cpu().numpy())
        self.measure.append(ratios)

    def compute_(self, **kwargs):
        self.metric.append(np.mean(self.measure))


class LMPolicyProbsRatio(Metric):
    '''to monitor the difference between the proba given by the lm for the words choosen and the probas given by the policy.'''

    def __init__(self, agent, train_test):
        Metric.__init__(self, agent, train_test, "lm_policy_probs_ratio", "scalar")

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
        Metric.__init__(self, agent, train_test, "action_probs", "scalar")

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
        Metric.__init__(self, agent, train_test, "action_probs_truncated", "scalar")

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
        Metric.__init__(self, agent, train_test, "action_probs_lm", "scalar")

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


metrics = {"return": Return, "valid_actions": VAMetric, "size_valid_actions": SizeVAMetric,
           "dialog": DialogMetric, "dialogimage": DialogImageMetric,
           "ppl": PPLMetric, "ppl_dialog_lm": PPLDialogfromLM, "bleu": BleuMetric,
           "ttr_question": TTRQuestionMetric, "sum_probs": SumProbsOverTruncated, "true_word_rank": TrueWordRankLM,
           "true_word_prob": TrueWordProbLM}
metrics_to_tensorboard = ["return", "size_valid_actions"]
