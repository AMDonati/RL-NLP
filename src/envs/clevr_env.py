import os
from collections import namedtuple

import gym
import numpy as np
import torch

from RL_toolbox.reward import rewards
from data_provider.CLEVR_Dataset import CLEVR_Dataset


class ClevrEnv(gym.Env):
    """Clevr Env"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, max_len, reward_type="levenshtein",
                 reward_path=None,
                 debug_len_vocab=None, max_samples=None, debug=False):
        super(ClevrEnv, self).__init__()
        self.data_path = data_path
        h5_questions_path = os.path.join(data_path, 'train_questions.h5')
        h5_feats_path = os.path.join(data_path, 'train_features.h5')
        vocab_path = os.path.join(data_path, 'vocab.json')
        # self.debug_true_questions = torch.randint(0,debug_len_vocab, (2,))
        self.debug = debug
        self.clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                           h5_feats_path=h5_feats_path,
                                           vocab_path=vocab_path,
                                           max_samples=max_samples)

        # num_tokens = self.clevr_dataset.len_vocab
        # feats_shape = self.clevr_dataset.feats_shape
        SOS_idx = self.clevr_dataset.vocab_questions["<SOS>"]
        EOS_idx = self.clevr_dataset.vocab_questions["<EOS>"]

        Special_Tokens = namedtuple('Special_Tokens', ('SOS_idx', 'EOS_idx'))
        self.special_tokens = Special_Tokens(SOS_idx, EOS_idx)
        self.State = namedtuple('State', ('text', 'img'))
        self.Episode = namedtuple('Episode',
                                  ('img_idx', 'img_feats', 'GD_questions', 'closest_question', 'dialog', 'rewards'))
        self.max_len = max_len
        # self.ref_questions = torch.randint(0, self.debug_len_vocab,
        #                                  (3, self.max_len)) if self.debug_len_vocab is not None else None
        # self.reset()

        self.reward_func = rewards[reward_type](reward_path)
        self.step_idx = 0
        self.state, self.dialog = None, None
        self.ref_questions, self.ref_questions_decoded = None, None
        self.img_idx, self.img_feats = None, None

    def step(self, action):
        action = torch.tensor(action).view(1, 1)
        self.state = self.State(torch.cat([self.state.text, action], dim=1), self.state.img)
        question = self.clevr_dataset.idx2word(self.state.text.numpy()[0])
        done = True if action.item() == self.special_tokens.EOS_idx or self.step_idx == (self.max_len - 1) else False
        # question = preprocess_final_state(state_text=self.state.text, dataset=self.clevr_dataset,
        #                                  EOS_idx=self.special_tokens.EOS_idx)
        reward, closest_question = self.reward_func.get(question=question,
                                                        ep_questions_decoded=self.ref_questions_decoded) if done else (
            0, None)
        self.step_idx += 1
        if done:
            self.dialog = question
            print(question)
        return self.state, (reward, closest_question), done, {}

    def reset(self):
        self.img_idx = np.random.randint(0, len(self.clevr_dataset))
        self.img_idx = 0  # for debugging.
        self.ref_questions = self.clevr_dataset.get_questions_from_img_idx(self.img_idx)  # shape (10, 45)
        self.ref_questions = self.ref_questions[:10, :self.max_len]  # shape (10, 45)

        # if self.debug:
        # self.ref_questions = torch.tensor([[7, 8, 10, 12, 14]])
        self.ref_questions_decoded = [
            self.clevr_dataset.idx2word(question, clean=True)
            for question in self.ref_questions.numpy()]
        print("Questions : {}".format(self.ref_questions_decoded))
        # self.ref_questions_decoded = [self.ref_questions_decoded[0]]  # FOR DEBUGGING.
        self.img_feats = self.clevr_dataset.get_feats_from_img_idx(self.img_idx)  # shape (1024, 14, 14)
        self.state = self.State(torch.LongTensor([self.special_tokens.SOS_idx]).view(1, 1), self.img_feats.unsqueeze(0))
        self.step_idx = 0
        self.dialog = None
        return self.state

    def get_reduced_action_space(self):
        assert self.ref_questions_decoded is not None
        reduced_vocab = [q.split() for q in self.ref_questions_decoded]
        reduced_vocab = [i for l in reduced_vocab for i in l]
        reduced_vocab = list(set(reduced_vocab))
        unique_tokens = self.clevr_dataset.word2idx(seq_tokens=reduced_vocab)
        dict_tokens = dict(zip([i for i in range(len(unique_tokens))], unique_tokens))
        return dict_tokens, reduced_vocab

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    env = ClevrEnv(data_path="../../data", max_len=5, max_samples=20)
    state = env.reset()
    dict_tokens, reduced_vocab = env.get_reduced_action_space()
    print(dict_tokens)
    print(list(dict_tokens.values()))
    print('len', len(dict_tokens))
    act_idx = 5
    action = dict_tokens[5]
    print(act_idx, action)
