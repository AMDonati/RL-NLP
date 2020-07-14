import json

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# TODO: add the intermediate reward (diff of R(t+1)-R(t))


class Reward:
    def __init__(self, path):
        self.path = path

    def get(self, question, ep_questions_decoded):
        pass


class Cosine(Reward):
    def __init__(self, path):
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


class Levenshtein(Reward):
    def __init__(self, correct_vocab=False, path=None):
        Reward.__init__(self, path)
        self.correct_vocab = correct_vocab

    def get(self, question, ep_questions_decoded, step_idx, done=False):
        if question is None:
            return 0., "N/A"
        else:
            distances = [nltk.edit_distance(question.split(), true_question.split()) / (
                max(len(question.split()), len(true_question.split()))) for true_question in
                         ep_questions_decoded]
            similarities = [1 - dist for dist in distances]
            sim_question_idx = np.asarray(similarities).argmax()
            closest_question = ep_questions_decoded[sim_question_idx]

            if self.correct_vocab:
                rew_vocab = self.get_rew_vocab(question, closest_question)
                reward = (1 - 0.1) * max(similarities) + 0.1 * rew_vocab
            else:
                reward = max(similarities)

            return reward, closest_question

    def get_rew_vocab(self, question, closest_question):
        vocab_ref_question = closest_question.split()
        vocab_question = question.split()
        intersect = list(set(vocab_ref_question).intersection(vocab_question))
        return len(intersect) / len(vocab_ref_question)


class CorrectVocab(Reward):
    def __init__(self, path):
        Reward.__init__(self, path=None)

    def get(self, question, ep_questions_decoded):
        if question is None or len(question.split()) == 0:
            return 0.
        else:
            vocab_ref_questions, len_questions = self.get_vocab(ep_questions_decoded)
            vocab_question, _ = self.get_vocab([question])
            intersect = list(set(vocab_ref_questions).intersection(vocab_question))
            return len(intersect) / max(len_questions)

    @staticmethod
    def get_vocab(questions):
        vocab = [q.split() for q in questions]
        len_questions = [len(q) for q in vocab]
        vocab = [i for l in vocab for i in l]
        vocab = list(set(vocab))
        return vocab, len_questions


class CombinedReward(Reward):
    def __init__(self, reward_func_1, reward_func_2, alpha, path):
        self.reward_func_1 = rewards[reward_func_1]()
        self.reward_func_2 = rewards[reward_func_2]()
        self.alpha = alpha
        Reward.__init__(self, path)

    def get(self, question, ep_questions_decoded):
        rew = (1 - self.alpha) * self.reward_func_1.get(question=question, ep_questions_decoded=ep_questions_decoded)[0] \
              + self.alpha * self.reward_func_2.get(question=question, ep_questions_decoded=ep_questions_decoded)
        return rew


class PerWord(Reward):
    def __init__(self, path=None):
        Reward.__init__(self, path)

    def get(self, question, ep_questions_decoded):
        similarities = np.asarray(
            [self.compare(question.split(), true_question.split()) for true_question in ep_questions_decoded])
        return max(similarities), ep_questions_decoded[similarities.argmax()]

    @staticmethod
    def compare(question, true_question):
        return sum([question[i + 1] == true_question[i] for i in range(len(true_question)) if
                    i < len(question) - 1])


class Levenshtein_(Reward):
    def __init__(self, path=None):
        Reward.__init__(self, path)
        self.type = "episode"

    def get(self, question, ep_questions_decoded, step_idx, done=False):
        if question is None:
            return 0., "N/A"
        distances = np.array([nltk.edit_distance(question.split()[1:], true_question.split()) for true_question in
                              ep_questions_decoded])
        reward = -min(distances) if done else 0
        return reward, ep_questions_decoded[distances.argmin()]


class Differential(Reward):
    def __init__(self, reward_function, path=None):
        Reward.__init__(self, path)
        self.type = "step"
        self.reward_function = reward_function
        self.last_reward = None

    def get(self, question, ep_questions_decoded, step_idx, done=False):
        if step_idx == 0:
            self.last_reward, _ = self.reward_function.get("", ep_questions_decoded, step_idx=step_idx, done=True)
        reward, closest_question = self.reward_function.get(question, ep_questions_decoded, step_idx=step_idx,
                                                            done=True)
        diff_reward = reward - self.last_reward
        self.last_reward = reward
        return diff_reward, closest_question


rewards = {"cosine": Cosine, "levenshtein": Levenshtein, "levenshtein_": Levenshtein_, "per_word": PerWord,
           "correct_vocab": CorrectVocab, "combined": CombinedReward}


def get_dummy_reward(next_state_text, ep_questions, EOS_idx):
    next_state_text = next_state_text[:, 1:]  # removing sos token.
    if next_state_text[:, -1] == EOS_idx:  # remove <EOS> token if needed.
        next_state_text = next_state_text[:, :-1]
    # trunc state:
    state_len = next_state_text.size(1)
    if state_len == 0:
        # print('no final reward...')
        return 0.
    else:
        # max_len = ep_questions.size(1)
        # if state_len < max_len:
        #     next_state_text = torch.cat([next_state_text, next_state_text.new_zeros(1, max_len - state_len)], dim=1)
        # assert max_len == next_state_text.size(1)
        ep_questions = ep_questions[:, :state_len]
        next_state_text = next_state_text.repeat(ep_questions.size(0), 1)
        mask_same_tokens = next_state_text == ep_questions
        reward = mask_same_tokens.sum(dim=-1).max().numpy()
        reward = reward / state_len
    return reward


if __name__ == '__main__':
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

    # --------------------------- CorrectVocab reward -----------------------------------------------------------------------------
    print("correct vocab reward...")

    reward_func = rewards["correct_vocab"](path=None)
    str_1 = "is it blue ?"
    str_2 = "is it blue ?"
    print(str_1)
    print(str_2)
    # rew = reward_func.get(str_1, [str_2])
    print('reward', rew)

    str_1 = "blue it is ?"
    str_2 = "is it blue ?"
    print(str_1)
    print(str_2)
    rew = reward_func.get(str_1, [str_2, "red is there a black ball ?"])
    print('reward', rew)

    str_1 = "red it is ?"
    str_2 = "is it blue ?"
    print(str_1)
    print(str_2)
    rew = reward_func.get(str_1, [str_2])
    print('reward', rew)

    str_1 = "red that object"
    str_2 = "is it blue"
    print(str_1)
    print(str_2)
    rew = reward_func.get(str_1, [str_2])
    print('reward', rew)

    # ----- lv reward with correct vocab ---------------------------------------------------------------------------------------
    print("lv reward with vocab...")
    reward_func = rewards["levenshtein"](path=None)
    reward_func_vocab = rewards["levenshtein"](path=None, correct_vocab=True)
    ref_questions = ["is it blue", "red is there a black ball"]
    str = "blue it is"
    print('question', str)
    print('ref_questions', ref_questions)
    print('reward w/o vocab', reward_func.get(str, ref_questions))
    print('rew with vocab', reward_func_vocab.get(str, ref_questions))
    str = "blue red it"
    print('question', str)
    print('ref_questions', ref_questions)
    print('reward w/o vocab', reward_func.get(str, ref_questions))
    print('rew with vocab', reward_func_vocab.get(str, ref_questions))
    str = "is it blue"
    print('question', str)
    print('ref_questions', ref_questions)
    print('reward w/o vocab', reward_func.get(str, ref_questions))
    print('rew with vocab', reward_func_vocab.get(str, ref_questions))
    # --------------- Combined Reward ----------------------------------------------------------------------------------------------
    # print("combined reward...")

    # reward_func = rewards["combined"](reward_func_1="levenshtein",
    # reward_func_2="correct_vocab", alpha=0.1, path=None)

# TODO: code a reward for taking in account the good words.
# TODO: editing distance that rewards correctly the length of question.
