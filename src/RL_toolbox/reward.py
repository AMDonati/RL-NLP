import json
import time
import nltk
import numpy as np
import pandas as pd
import torch
from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import torch.nn.functional as F
import logging

try:
    from vilbert.task_utils import compute_score_with_logits
    from vilbert.vilbert import VILBertForVLTasks, BertConfig
except ImportError:
    print("VILBERT NOT IMPORTED!!")

try:
    from vr.utils import load_execution_engine, load_program_generator
except ImportError:
    print("VR NOT IMPORTED!!")

logger = logging.getLogger()


def get_vocab(key, vocab_path):
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)[key]
    return vocab


def get_smoothing_function(sf_id):
    if sf_id == 0:
        smoothing_function = SmoothingFunction().method0
    elif sf_id == 1:
        smoothing_function = SmoothingFunction().method1
    elif sf_id == 2:
        smoothing_function = SmoothingFunction().method2
    elif sf_id == 3:
        smoothing_function = SmoothingFunction().method3
    elif sf_id == 4:
        smoothing_function = SmoothingFunction().method4
    elif sf_id == 5:
        smoothing_function = SmoothingFunction().method5
    elif sf_id == 6:
        smoothing_function = SmoothingFunction().method6
    elif sf_id == 7:
        smoothing_function = SmoothingFunction().method7
    return smoothing_function


def get_weights_bleu_score(n_gram=4):
    if n_gram == 2:
        weights = [0.5, 0.5]
    elif n_gram == 3:
        weights = [1 / 3, 1 / 3, 1 / 3]
    elif n_gram == 4:
        weights = [0.25, 0.25, 0.25, 0.25]
    return weights


class Reward:
    def __init__(self, path, vocab=None, dataset=None, env=None):
        self.path = path
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if env is None else env.device

    def get(self, question, ep_questions_decoded):
        pass


class Cosine(Reward):
    def __init__(self, path, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        with open(path) as json_file:
            data = json.load(json_file)
        df = pd.read_json(json.dumps(data["questions"]))
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(df.question)

    def get(self, question, ep_questions_decoded):
        data = np.append(np.array([question]), np.array(ep_questions_decoded))
        X = self.vectorizer.transform(data)
        S = cosine_similarity(X[0:1], X[1:])
        rew = max(S[0])
        return rew


class LevenshteinNorm(Reward):
    def __init__(self, correct_vocab=False, path=None, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        self.correct_vocab = correct_vocab

    def get(self, question, ep_questions_decoded, step_idx, done=False, real_answer="", state=None):
        if question is None:
            return 0., "N/A", None
        else:
            distances = np.array([nltk.edit_distance(question.split(), true_question.split()) for true_question in
                                  ep_questions_decoded])
            distance = min(distances)
            closest_question = ep_questions_decoded[distances.argmin()]
            distance_norm = distance / (max(len(question.split()), len(closest_question.split())))
            reward = -distance_norm if done else 0
            return reward, closest_question, None


class Levenshtein_(Reward):
    def __init__(self, path=None, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        self.type = "episode"

    def get(self, question, ep_questions_decoded, step_idx, done=False, real_answer="", state=None):
        if question is None:
            return 0., "N/A"
        distances = np.array([nltk.edit_distance(question.split(), true_question.split()) for true_question in
                              ep_questions_decoded])
        reward = -min(distances) if done else 0
        return reward, ep_questions_decoded[distances.argmin()], None


class Bleu_sf2(Reward):
    def __init__(self, sf_id=2, n_gram=4, path=None, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        self.type = "episode"
        self.sf_id = sf_id
        self.n_gram = n_gram
        self.smoothing_function = get_smoothing_function(sf_id)
        self.weights = get_weights_bleu_score(n_gram)

    def get(self, question, ep_questions_decoded, step_idx=None, done=False, real_answer="", state=None):
        if not done:
            return 0, "N/A", None
        normalize_function = lambda x: x.replace("?", " ?").split()
        ep_questions_decoded_normalized = [normalize_function(question) for question in ep_questions_decoded]
        reward = sentence_bleu(ep_questions_decoded_normalized, normalize_function(question),
                               smoothing_function=self.smoothing_function, weights=self.weights)
        scores = [sentence_bleu([ref], normalize_function(question), smoothing_function=self.smoothing_function,
                                weights=self.weights) for ref in
                  ep_questions_decoded_normalized]
        closest_question = ep_questions_decoded[np.array(scores).argmax()]
        return reward, closest_question, None


class Bleu_sf4(Reward):
    def __init__(self, sf_id=4, n_gram=4, path=None, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        self.type = "episode"
        self.sf_id = sf_id
        self.n_gram = n_gram
        self.smoothing_function = get_smoothing_function(sf_id)
        self.weights = get_weights_bleu_score(n_gram)

    def get(self, question, ep_questions_decoded, step_idx=None, done=False, real_answer="", state=None):
        if not done:
            return 0, "N/A", None
        normalize_function = lambda x: x.replace("?", " ?").split()
        ep_questions_decoded_normalized = [normalize_function(question) for question in ep_questions_decoded]
        reward = sentence_bleu(ep_questions_decoded_normalized, normalize_function(question),
                               smoothing_function=self.smoothing_function, weights=self.weights)
        scores = [sentence_bleu([ref], normalize_function(question), smoothing_function=self.smoothing_function,
                                weights=self.weights) for ref in
                  ep_questions_decoded_normalized]
        closest_question = ep_questions_decoded[np.array(scores).argmax()]
        return reward, closest_question, None


class Differential(Reward):
    def __init__(self, reward_function, path=None, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        self.type = "step"
        self.reward_function = reward_function
        self.last_reward = None

    def get(self, question, ep_questions_decoded, step_idx, done=False, real_answer="", state=None):
        if step_idx == 0:
            self.last_reward, _, _ = self.reward_function.get("", ep_questions_decoded, step_idx=step_idx, done=True, real_answer=real_answer, state=state)
        reward, closest_question, pred_answer = self.reward_function.get(question, ep_questions_decoded,
                                                                         step_idx=step_idx,
                                                                         done=True, real_answer=real_answer, state=state)
        diff_reward = reward - self.last_reward
        self.last_reward = reward
        return diff_reward, closest_question, pred_answer


class VQAAnswer(Reward):
    def __init__(self, path=None, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        self.type = "episode"
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if env is None else env.device
        self.execution_engine, ee_kwargs = load_execution_engine(path)
        self.execution_engine.to(self.device)
        self.execution_engine.eval()
        self.program_generator, pg_kwargs = load_program_generator(path)
        self.program_generator.to(self.device)
        self.program_generator.eval()
        self.vocab = vocab
        self.dataset = dataset
        self.vocab_questions_vqa = get_vocab('question_token_to_idx', self.vocab)
        # self.vocab_questions_vqa.update({"<pad>": 0, "<sos>": 1, "<eos>": 2})
        self.trad_dict = {value: self.vocab_questions_vqa[key] for key, value in self.dataset.vocab_questions.items() if
                          key in self.vocab_questions_vqa}

    def trad(self, state):
        idx_vqa = [self.trad_dict[idx] for idx in state.text.squeeze().cpu().numpy() if idx in self.trad_dict]
        idx_vqa.insert(0, 1)  # add SOS token.
        idx_vqa.append(2)  # add EOS token.
        return torch.tensor(idx_vqa).unsqueeze(dim=0)

    def get(self, question, ep_questions_decoded, step_idx, done=False, real_answer="", state=None):
        if not done:
            return 0, "N/A", None
        with torch.no_grad():
            question = self.trad(state).to(self.device)
            programs_pred = self.program_generator(question)
            scores = self.execution_engine(state.img.to(self.device), programs_pred)
            _, preds = scores.data.cpu().max(1)
            reward = (preds == real_answer).sum().item()
        return reward, "N/A", preds


class VILBERT(Reward):
    def __init__(self, path=None, vocab=None, dataset=None, env=None):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') if env is None else env.device
        self.dataset = env.dataset if env is not None else None
        self.task_id = 1
        self.env = env
        config = BertConfig.from_json_file(vocab)
        self.model = VILBertForVLTasks.from_pretrained(path, config=config, num_labels=1)
        self.reduced_answers = self.env.reduced_answers

    def get_preds(self, question=None, ep_questions_decoded=None):
        (features,
         spatials,
         image_mask,
         real_question,
         target,
         real_input_mask,
         real_segment_ids,
         co_attention_mask,
         question_id) = self.dataset.get_data_for_ViLBERT(self.env.env_idx)
        # logger.info(real_question)
        if question is None:
            question = ep_questions_decoded[0]
        #logger.info(question)
        encoded_question = self.dataset.reward_tokenizer.encode(question)
        encoded_question = self.dataset.reward_tokenizer.add_special_tokens_single_sentence(encoded_question)

        segment_ids, input_mask, encoded_question = self.dataset.get_masks_for_tokens(encoded_question)
        segment_ids, input_mask, encoded_question = torch.tensor(segment_ids), torch.tensor(input_mask), torch.tensor(
            encoded_question).view(-1)
        task_tokens = encoded_question.new().resize_(encoded_question.size(0), 1).fill_(int(self.task_id))
        vil_prediction, vil_prediction_gqa, vil_logit, vil_binary_prediction, vil_tri_prediction, vision_prediction, vision_logit, linguisic_prediction, linguisic_logit, _ = self.model(
            encoded_question.unsqueeze(dim=0),
            features.unsqueeze(dim=0),
            spatials.unsqueeze(dim=0),
            segment_ids.unsqueeze(dim=0),
            input_mask.unsqueeze(dim=0),
            image_mask.unsqueeze(dim=0),
            co_attention_mask.unsqueeze(dim=0),
            task_tokens
        )

        return vil_prediction, target

    def get_reward(self, sorted_logits, vil_prediction, target, ranks, rank):
        reward = compute_score_with_logits(vil_prediction, target.unsqueeze(dim=0), device=self.device)
        reward = reward.sum().item()
        return reward

    def get(self, question, ep_questions_decoded, step_idx, done=False, real_answer="", state=None, entry=None):
        if not done:
            return 0, "N/A", None
        vil_prediction, target = self.get_preds(question)
        if self.reduced_answers:
            mask = torch.ones_like(vil_prediction) * float("-Inf")
            mask[:, self.dataset.reduced_answers.squeeze()] = vil_prediction[:,
                                                              self.dataset.reduced_answers.squeeze()]
            vil_prediction = mask
        sorted_logits, sorted_indices = torch.sort(vil_prediction, descending=True)
        ranks = (sorted_indices.squeeze()[..., None] == (target != 0).nonzero().squeeze()).any(-1).nonzero()
        rank = ranks.min().item()
        reward = self.get_reward(sorted_logits, vil_prediction, target, ranks, rank, ep_questions_decoded)
        return reward, "N/A", vil_prediction.argmax()


class VILBERT_proba(VILBERT):
    def __init__(self, path=None, vocab=None, dataset=None, env=None):
        super().__init__(path=path, vocab=vocab, dataset=dataset, env=env)

    def get_reward(self, sorted_logits, vil_prediction, target, ranks, rank, ep_question_decoded):
        sorted_probs = F.softmax(sorted_logits, dim=1)
        reward = sorted_probs[:, rank]
        return reward


class VILBERT_rank(VILBERT):
    def __init__(self, path=None, vocab=None, dataset=None, env=None):
        super().__init__(path=path, vocab=vocab, dataset=dataset, env=env)

    def get_reward(self, sorted_logits, vil_prediction, target, ranks, rank, ep_question_decoded):
        reward = -np.log(rank + 1)
        return reward


class VILBERT_rank2(VILBERT):
    def __init__(self, path=None, vocab=None, dataset=None, env=None):
        super().__init__(path=path, vocab=vocab, dataset=dataset, env=env)

    def get_reward(self, sorted_logits, vil_prediction, target, ranks, rank, ep_question_decoded):
        reward = np.exp((1 - rank) / 2)
        return reward


class VILBERT_rank_DCG2(VILBERT):
    def __init__(self, path=None, vocab=None, dataset=None, env=None):
        super().__init__(path=path, vocab=vocab, dataset=dataset, env=env)

    def get_reward(self, sorted_logits, vil_prediction, target, ranks, rank, ep_question_decoded):
        # sorted_probs = F.softmax(sorted_logits, dim=1)
        # rank_prob = sorted_probs[:, rank]
        reward = 1 / np.log2(2 + rank)
        return reward


class VILBERT_rank_DCG(VILBERT):
    def __init__(self, path=None, vocab=None, dataset=None, env=None):
        super().__init__(path=path, vocab=vocab, dataset=dataset, env=env)
        self.normalize = False

    def get_reward(self, sorted_logits, vil_prediction, target, ranks, rank, ep_question_decoded):
        # getting the  vil prediction fo the true question
        true_vil_prediction, target = self.get_preds(ep_questions_decoded=ep_question_decoded)
        true_probs = F.softmax(true_vil_prediction, dim=1)
        importances = true_probs * 100
        true_sorted_importances, true_sorted_indices = torch.sort(importances, descending=True)
        range_ = torch.arange(0, true_sorted_indices.size(1))
        ref_dcg = true_sorted_importances / torch.log2(2. + range_)
        ref_dcg = ref_dcg.sum()

        _, sorted_indices = torch.sort(vil_prediction, descending=True)
        importances_ = torch.gather(importances, 1, sorted_indices)
        dcg = importances_ / torch.log2(2. + range_)
        dcg = dcg.sum()
        if self.normalize:
            dcg = dcg / ref_dcg
        return dcg.item()


class VILBERT_rank_NDCG(VILBERT_rank_DCG):
    def __init__(self, path=None, vocab=None, dataset=None, env=None):
        super().__init__(path=path, vocab=vocab, dataset=dataset, env=env)
        self.normalize = True


# ---------------------------------------------other Bleu variants------------------------------------------------------
class Bleu_sf7(Reward):
    def __init__(self, sf_id=7, n_gram=4, path=None, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        self.type = "episode"
        self.sf_id = sf_id
        self.n_gram = n_gram
        self.smoothing_function = get_smoothing_function(sf_id)
        self.weights = get_weights_bleu_score(n_gram)

    def get(self, question, ep_questions_decoded, step_idx=None, done=False, real_answer="", state=None):
        if not done:
            return 0, "N/A", None
        normalize_function = lambda x: x.replace("?", " ?").split()
        ep_questions_decoded_normalized = [normalize_function(question) for question in ep_questions_decoded]
        reward = sentence_bleu(ep_questions_decoded_normalized, normalize_function(question),
                               smoothing_function=self.smoothing_function, weights=self.weights)
        scores = [sentence_bleu([ref], normalize_function(question), smoothing_function=self.smoothing_function,
                                weights=self.weights) for ref in
                  ep_questions_decoded_normalized]
        closest_question = ep_questions_decoded[np.array(scores).argmax()]
        return reward, closest_question, None


class Bleu_sf3(Reward):
    def __init__(self, sf_id=3, n_gram=4, path=None, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        self.type = "episode"
        self.sf_id = sf_id
        self.n_gram = n_gram
        self.smoothing_function = get_smoothing_function(sf_id)
        self.weights = get_weights_bleu_score(n_gram)

    def get(self, question, ep_questions_decoded, step_idx=None, done=False, real_answer="", state=None):
        if not done:
            return 0, "N/A", None
        normalize_function = lambda x: x.replace("?", " ?").split()
        ep_questions_decoded_normalized = [normalize_function(question) for question in ep_questions_decoded]
        reward = sentence_bleu(ep_questions_decoded_normalized, normalize_function(question),
                               smoothing_function=self.smoothing_function, weights=self.weights)
        scores = [sentence_bleu([ref], normalize_function(question), smoothing_function=self.smoothing_function,
                                weights=self.weights) for ref in
                  ep_questions_decoded_normalized]
        closest_question = ep_questions_decoded[np.array(scores).argmax()]
        return reward, closest_question, None


class Bleu_sf0(Reward):
    def __init__(self, sf_id=0, n_gram=4, path=None, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        self.type = "episode"
        self.sf_id = sf_id
        self.n_gram = n_gram
        self.smoothing_function = get_smoothing_function(sf_id)
        self.weights = get_weights_bleu_score(n_gram)

    def get(self, question, ep_questions_decoded, step_idx=None, done=False, real_answer="", state=None):
        if not done:
            return 0, "N/A", None
        normalize_function = lambda x: x.replace("?", " ?").split()
        ep_questions_decoded_normalized = [normalize_function(question) for question in ep_questions_decoded]
        reward = sentence_bleu(ep_questions_decoded_normalized, normalize_function(question),
                               smoothing_function=self.smoothing_function, weights=self.weights)
        scores = [sentence_bleu([ref], normalize_function(question), smoothing_function=self.smoothing_function,
                                weights=self.weights) for ref in
                  ep_questions_decoded_normalized]
        closest_question = ep_questions_decoded[np.array(scores).argmax()]
        return reward, closest_question, None


class Bleu_sf1(Reward):
    def __init__(self, sf_id=1, n_gram=4, path=None, vocab=None, dataset=None, env=None):
        Reward.__init__(self, path)
        self.type = "episode"
        self.sf_id = sf_id
        self.n_gram = n_gram
        self.smoothing_function = get_smoothing_function(sf_id)
        self.weights = get_weights_bleu_score(n_gram)

    def get(self, question, ep_questions_decoded, step_idx=None, done=False, real_answer="", state=None):
        if not done:
            return 0, "N/A", None
        normalize_function = lambda x: x.replace("?", " ?").split()
        ep_questions_decoded_normalized = [normalize_function(question) for question in ep_questions_decoded]
        reward = sentence_bleu(ep_questions_decoded_normalized, normalize_function(question),
                               smoothing_function=self.smoothing_function, weights=self.weights)
        scores = [sentence_bleu([ref], normalize_function(question), smoothing_function=self.smoothing_function,
                                weights=self.weights) for ref in
                  ep_questions_decoded_normalized]
        closest_question = ep_questions_decoded[np.array(scores).argmax()]
        return reward, closest_question, None


rewards = {"cosine": Cosine, "levenshtein": Levenshtein_, "lv_norm": LevenshteinNorm, "vqa": VQAAnswer,
           "bleu": Bleu_sf2,
           "bleu_sf0": Bleu_sf0, "bleu_sf1": Bleu_sf1, "bleu_sf2": Bleu_sf2, "bleu_sf3": Bleu_sf3, "bleu_sf4": Bleu_sf4,
           "bleu_sf7": Bleu_sf7,
           "vilbert": VILBERT, "vilbert_rank": VILBERT_rank, "vilbert_rank2": VILBERT_rank2,
           "vilbert_proba": VILBERT_proba, "vilbert_dcg": VILBERT_rank_DCG, "vilbert_dcg2": VILBERT_rank_DCG2,
           "vilbert_ndcg": VILBERT_rank_NDCG}

if __name__ == '__main__':
    print("testing of BLEU score with sf7 smoothing function")
    reward_sf7 = rewards["bleu_sf7"]()
    rew_1 = reward_sf7.get(question="The cat is on the mat", ep_questions_decoded=["The cat is on the mat"], done=True)
    print(rew_1)
    rew_0 = reward_sf7.get(question="The cat is on the mat", ep_questions_decoded=["the the the the the the"],
                           done=True)
    print(rew_0)
    rew_2 = reward_sf7.get(question="What kinds of birds are flying",
                           ep_questions_decoded=["What kinds of birds are flying"], done=True)
    print(rew_2)

    print("testing of BLEU score with sf4 smoothing function")
    reward_sf7 = rewards["bleu"]()
    rew_1 = reward_sf7.get(question="The cat is on the mat", ep_questions_decoded=["The cat is on the mat"],
                           step_idx=None, done=True)
    print(rew_1)
    # rew_0 = reward_sf7.get(question="The cat is on the mat", ep_questions_decoded=["What kinds of birds are flying"], step_idx=None, done=True)
    # print(rew_0)
    rew_0 = reward_sf7.get(question="The cat is on the mat",
                           ep_questions_decoded=["the the the the the the"], done=True, step_idx=None)
    print(rew_0)

    reward_func = rewards["cosine"](path="../../data/CLEVR_v1.0/temp/50000_20000_samples_old/train_questions.json")
    rew = reward_func.get("is it blue ?", ["is it red ?", "where is it ?"])
    print("reward {} cosine".format(rew))
    rew = reward_func.get("is it blue ?", ["is it blue ?", "where is it ?"])
    print("reward {} cosine".format(rew))

    print("levenshtein rewards...")

    reward_func = rewards["levenshtein"](path=None)
    reward_func_ = rewards["levenshtein_"](path=None)

    rew = reward_func.get("is it blue ?", ["is it red ?", "where is it ?"])
    print("reward {} levenshtein".format(rew))

    rew = reward_func.get("is it blue ?", ["is it blue ?", "where is it ?"])
    print("reward {} levenshtein".format(rew))

    str_1 = "is it blue and tiny ?"
    str_2 = "is it blue ?"
    print('question', str_1)
    print('true question', str_2)

    rew_norm_pos, sim_q = reward_func.get(str_1, [str_2])
    rew, _ = reward_func_.get(str_1, [str_2])
    print('rew norm positive', rew_norm_pos)
    print('rew not norm', rew)

    str_1 = "is it red ?"
    str_2 = "is it blue ?"
    print('question', str_1)
    print('true question', str_2)

    rew_norm_pos, sim_q = reward_func.get(str_1, [str_2])
    rew, _ = reward_func_.get(str_1, [str_2])
    print('rew norm positive', rew_norm_pos)
    print('rew not norm', rew)

    print('checking rew with an empty string...')
    rew_norm_pos, sim_q = reward_func.get("", [str_2])
    print('rew norm positive', rew_norm_pos)
    print('similar question', sim_q)

    str_1 = "blue red it"
    str_2 = "is it blue"
    print('question', str_1)
    print('true question', str_2)

    rew_norm_pos, sim_q = reward_func.get(str_1, [str_2])
    print('rew norm positive', rew_norm_pos)
