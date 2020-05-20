import json

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


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
    def __init__(self, path):
        Reward.__init__(self, path)

    def get(self, question, ep_questions_decoded):
        if question is None:
            return 0., None
        else:
            distances = [nltk.edit_distance(question.split(), true_question.split()) / (
                max(len(question.split()), len(true_question.split()))) for true_question in
                         ep_questions_decoded]
            similarities = [1 - dist for dist in distances]
            sim_question_idx = np.asarray(similarities).argmax()
            closest_question = ep_questions_decoded[sim_question_idx]

            return max(similarities), closest_question


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

    def get(self, question, ep_questions_decoded):
        if question is None:
            print("no")
        distances = np.array([nltk.edit_distance(question.split()[1:], true_question.split()) for true_question in
                              ep_questions_decoded])
        self.last_reward = -min(distances)
        return self.last_reward, ep_questions_decoded[distances.argmin()]

    def get_diff(self, question, ep_questions_decoded):
        prev_reward = self.last_reward
        reward = self.get(question, ep_questions_decoded)
        return reward - prev_reward


rewards = {"cosine": Cosine, "levenshtein": Levenshtein, "levenshtein_": Levenshtein_, "per_word": PerWord}


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

    reward_func = rewards["levenshtein"](path="../../data/CLEVR_v1.0/temp/50000_20000_samples_old/train_questions.json")
    rew = reward_func.get("is it blue ?", ["is it red ?", "where is it ?"])
    print("reward {} levenshtein".format(rew))

    rew = reward_func.get("is it blue ?", ["is it blue ?", "where is it ?"])
    print("reward {} levenshtein".format(rew))

    str_1 = "is it blue and tiny ?"
    str_2 = "is it blue ?"

    rew = reward_func.get_old(str_1, [str_2])
    rew_norm = rew / max(len(str_1.split()), len(str_2.split()))
    print('rew negative ', rew)
    print('rew norm negative', rew_norm)

    rew_norm_pos, sim_q = reward_func.get(str_1, [str_2])
    print('rew norm positive', rew_norm_pos)
    print('similar question', sim_q)

    str_1 = "is it red ?"
    str_2 = "is it blue ?"

    rew = reward_func.get_old(str_1, [str_2])
    rew_norm = rew / max(len(str_1.split()), len(str_2.split()))
    print('rew', rew)
    print('rew norm', rew_norm)

    rew_norm_pos, sim_q = reward_func.get(str_1, [str_2])
    print('rew norm positive', rew_norm_pos)
    print('similar question', sim_q)
