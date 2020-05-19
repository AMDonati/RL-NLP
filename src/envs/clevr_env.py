import os
from collections import namedtuple

import gym
import numpy as np
import torch

from data_provider.CLEVR_Dataset import CLEVR_Dataset
from train.reward import rewards


class ClevrEnv(gym.Env):
    """Clevr Env"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, max_len, reward_type="cosine",
                 reward_path="../../data/CLEVR_v1.0/temp/50000_20000_samples_old/train_questions.json",
                 debug_len_vocab=None):
        super(ClevrEnv, self).__init__()
        self.data_path = data_path
        h5_questions_path = os.path.join(data_path, 'train_questions.h5')
        h5_feats_path = os.path.join(data_path, 'train_features.h5')
        vocab_path = os.path.join(data_path, 'vocab.json')
        # self.debug_true_questions = torch.randint(0,debug_len_vocab, (2,))
        self.debug_len_vocab = debug_len_vocab
        self.clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                           h5_feats_path=h5_feats_path,
                                           vocab_path=vocab_path, debug_len_vocab=self.debug_len_vocab)

        # num_tokens = self.clevr_dataset.len_vocab
        # feats_shape = self.clevr_dataset.feats_shape
        SOS_idx = self.clevr_dataset.vocab_questions["<SOS>"]
        EOS_idx = self.clevr_dataset.vocab_questions["<EOS>"]

        Special_Tokens = namedtuple('Special_Tokens', ('SOS_idx', 'EOS_idx'))
        self.special_tokens = Special_Tokens(SOS_idx, EOS_idx)
        self.State = namedtuple('State', ('text', 'img'))
        self.Episode = namedtuple('Episode', (
            'img_idx', 'img_feats', 'GD_questions', 'dialog', 'rewards'))  # TODO: Build an Episode Dataset instead.
        self.step_idx = 0
        self.state = None
        self.max_len = max_len
        #self.ref_questions = torch.randint(0, self.debug_len_vocab,
         #                                  (3, self.max_len)) if self.debug_len_vocab is not None else None
        self.ref_questions=torch.tensor([[7, 8, 10, 12, 14]])
        self.ref_questions_decoded = None
        self.reset()

        self.reward_func = rewards[reward_type](reward_path)

    def step(self, action):
        action = torch.tensor(action).view(1, 1)
        self.state = self.State(torch.cat([self.state.text, action], dim=1), self.state.img)
        question = self.clevr_dataset.decode(self.state.text.numpy()[0])
        done = True if action.item() == self.special_tokens.EOS_idx or self.step_idx == (self.max_len - 1) else False
        if done:
            print(question)
        reward = self.reward_func.get(question=question, ep_questions_decoded=self.ref_questions_decoded) if done else 0
        # reward = self.reward_func.get(question=question, ep_questions_decoded=self.ref_questions_decoded)

        self.step_idx += 1
        return self.state, reward, done, {}

    def reset(self):
        img_idx = np.random.randint(0, len(self.clevr_dataset.img_idxs))
        img_idx = 10  # for debugging.

        # self.ref_questions = self.ref_questions[1:2]

        if self.debug_len_vocab is None:
            self.ref_questions = self.clevr_dataset.get_questions_from_img_idx(
                img_idx)  # shape (max_len - 1, 10) # used to compute the final reward of the episode.
            # self.ref_questions = torch.tensor(self.debug_true_questions)
        self.ref_questions_decoded = [
            self.clevr_dataset.decode(question).replace(" <PAD>", "")
            for question in self.ref_questions.numpy()]
        # self.ref_questions_decoded = ["Are Do"]
        print(self.ref_questions_decoded)
        img_feats = self.clevr_dataset.get_feats_from_img_idx(img_idx)  # shape (1024, 14, 14)
        self.state = self.State(torch.LongTensor([self.special_tokens.SOS_idx]).view(1, 1), img_feats.unsqueeze(0))
        self.step_idx = 0
        return self.state

    def render(self, mode='human', close=False):
        pass
