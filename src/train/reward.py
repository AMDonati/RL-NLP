import json

import nltk
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class Reward:
    def __init__(self, path=None):
        self.path = path
        self.last_reward = 0

    def get(self, question, ep_questions_decoded):
        pass


class Cosine(Reward):
    def __init__(self, path=None):
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
        return float(rew)


class PerWord(Reward):
    def __init__(self, path=None):
        Reward.__init__(self, path)

    def get(self, question, ep_questions_decoded):
        reward = max([self.compare(question.split(), true_question.split()) for true_question in ep_questions_decoded])
        return reward

    @staticmethod
    def compare(question, true_question):
        return sum([question[i + 1] == true_question[i] for i in range(len(true_question)) if
                    i < len(question)-1])


class Levenshtein(Reward):
    def __init__(self, path=None):
        Reward.__init__(self, path)

    def get(self, question, ep_questions_decoded):
        distances = [nltk.edit_distance(question.split()[1:], true_question.split()) for true_question in
                     ep_questions_decoded]
        self.last_reward = -min(distances)
        # self.last_reward/=(25-len(question))
        return self.last_reward

    def get_diff(self, question, ep_questions_decoded):
        prev_reward = self.last_reward
        reward = self.get(question, ep_questions_decoded)
        return reward - prev_reward


rewards = {"cosine": Cosine, "levenshtein": Levenshtein, "perword": PerWord}

if __name__ == '__main__':
    reward_func = rewards["cosine"](path="../../data/CLEVR_v1.0/temp/50000_20000_samples_old/train_questions.json")
    rew = reward_func.get("is it blue ?", ["is it red ?", "where is it ?"])
    print("reward {} cosine".format(rew))
    rew = reward_func.get("is it blue ?", ["is it blue ?", "where is it ?"])
    print("reward {} cosine".format(rew))

    reward_func = rewards["levenshtein"](path="../../data/CLEVR_v1.0/temp/50000_20000_samples_old/train_questions.json")
    rew = reward_func.get("is it blue ?", ["is it red ?", "where is it ?"])
    print("reward {} levenshtein".format(rew))

    rew = reward_func.get("is it blue ?", ["is it blue ?", "where is it ?"])
    print("reward {} levenshtein".format(rew))
