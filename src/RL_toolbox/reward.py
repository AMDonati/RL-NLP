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

    def get_old(self, question, ep_questions_decoded):
        distances = [nltk.edit_distance(question.split(), true_question.split()) for true_question in ep_questions_decoded]
        return -min(distances)

    def get(self, question, ep_questions_decoded):
        distances = [nltk.edit_distance(question.split(), true_question.split()) / (max(len(question.split()), len(true_question.split()))) for true_question in
                     ep_questions_decoded]
        similarities = [1-dist for dist in distances]
        return max(similarities)

rewards = {"cosine": Cosine, "levenshtein": Levenshtein}

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

    str_1 = "is it blue and tiny?"
    str_2 = "is it blue ?"

    rew = reward_func.get_old(str_1, [str_2])
    rew_norm = rew / max(len(str_1.split()), len(str_2.split()))
    print('rew negative ', rew)
    print('rew norm negative', rew_norm)

    rew_norm_pos = reward_func.get(str_1, [str_2])
    print('rew norm positive', rew_norm_pos)

    str_1 = "is it red?"
    str_2 = "is it blue ?"


    rew = reward_func.get_old(str_1, [str_2])
    rew_norm = rew / max(len(str_1.split()), len(str_2.split()))
    print('rew', rew)
    print('rew norm', rew_norm)

    rew_norm_pos = reward_func.get(str_1, [str_2])
    print('rew norm positive', rew_norm_pos)