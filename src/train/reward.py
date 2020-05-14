import json

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
        return max(S[0])


class Levenshtein(Reward):
    def __init__(self, path):
        Reward.__init__(self, path)

    def get(self, question, ep_questions_decoded):
        distances = [self.levenshtein(question, true_question) for true_question in ep_questions_decoded]
        return -min(distances)

    @staticmethod
    def levenshtein(seq1, seq2):
        size_x = len(seq1) + 1
        size_y = len(seq2) + 1
        matrix = np.zeros((size_x, size_y))
        for x in range(size_x):
            matrix[x, 0] = x
        for y in range(size_y):
            matrix[0, y] = y

        for x in range(1, size_x):
            for y in range(1, size_y):
                if seq1[x - 1] == seq2[y - 1]:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1],
                        matrix[x, y - 1] + 1
                    )
                else:
                    matrix[x, y] = min(
                        matrix[x - 1, y] + 1,
                        matrix[x - 1, y - 1] + 1,
                        matrix[x, y - 1] + 1
                    )
        return matrix[size_x - 1, size_y - 1]


rewards = {"cosine": Cosine, "levenshtein": Levenshtein}

if __name__ == '__main__':
    reward_func = rewards["cosine"](path="../../data/CLEVR_v1.0/temp/50000_20000_samples/train_questions.json")
    rew = reward_func.get("is it blue ?", ["is it red ?", "where is it ?"])
    print("reward {} cosine".format(rew))
    rew = reward_func.get("is it blue ?", ["is it blue ?", "where is it ?"])
    print("reward {} cosine".format(rew))

    reward_func = rewards["levenshtein"](path="../../data/CLEVR_v1.0/temp/50000_20000_samples/train_questions.json")
    rew = reward_func.get("is it blue ?", ["is it red ?", "where is it ?"])
    print("reward {} levenshtein".format(rew))

    rew = reward_func.get("is it blue ?", ["is it blue ?", "where is it ?"])
    print("reward {} levenshtein".format(rew))

