import os
from collections import namedtuple

import gym
import numpy as np
import torch

from RL_toolbox.RL_functions import preprocess_final_state
from RL_toolbox.reward import rewards
from data_provider.CLEVR_Dataset import CLEVR_Dataset
from preprocessing.text_functions import decode


class ClevrEnv(gym.Env):
    """Clevr Env"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, max_len, reward_type="cosine",
                 reward_path=None,
                 debug_len_vocab=None, max_samples=None):
        super(ClevrEnv, self).__init__()
        self.data_path = data_path
        h5_questions_path = os.path.join(data_path, 'train_questions.h5')
        h5_feats_path = os.path.join(data_path, 'train_features.h5')
        vocab_path = os.path.join(data_path, 'vocab.json')
        # self.debug_true_questions = torch.randint(0,debug_len_vocab, (2,))
        self.debug_len_vocab = debug_len_vocab
        self.clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                           h5_feats_path=h5_feats_path,
                                           vocab_path=vocab_path, debug_len_vocab=self.debug_len_vocab,
                                           max_samples=max_samples)

        # num_tokens = self.clevr_dataset.len_vocab
        # feats_shape = self.clevr_dataset.feats_shape
        SOS_idx = self.clevr_dataset.vocab_questions["<SOS>"]
        EOS_idx = self.clevr_dataset.vocab_questions["<EOS>"]

        Special_Tokens = namedtuple('Special_Tokens', ('SOS_idx', 'EOS_idx'))
        self.special_tokens = Special_Tokens(SOS_idx, EOS_idx)
        self.State = namedtuple('State', ('text', 'img'))
        self.Episode = namedtuple('Episode', ('img_idx', 'img_feats', 'GD_questions','closest_question', 'dialog', 'rewards'))
        self.max_len = max_len
        # self.ref_questions = torch.randint(0, self.debug_len_vocab,
        #                                  (3, self.max_len)) if self.debug_len_vocab is not None else None
        #self.ref_questions = torch.tensor([[7, 8, 10, 12, 14]])
        #self.ref_questions_decoded = None
        #self.reset()

        self.reward_func = rewards[reward_type](reward_path)
        self.step_idx = 0
        self.state, self.dialog = None, None
        self.ref_questions, self.ref_questions_decoded = None, None
        self.img_idx, self.img_feats = None, None


    def step(self, action):
        action = torch.tensor(action).view(1, 1)
        self.state = self.State(torch.cat([self.state.text, action], dim=1), self.state.img)
        question = self.clevr_dataset.decode(self.state.text.numpy()[0])
        done = True if action.item() == self.special_tokens.EOS_idx or self.step_idx == (self.max_len - 1) else False
        question = preprocess_final_state(state_text=self.state.text, dataset=self.clevr_dataset,
                                          EOS_idx=self.special_tokens.EOS_idx)
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
        self.ref_questions_decoded = [
            decode(question, idx_to_token=self.clevr_dataset.idx_to_token, stop_at_end=True).replace(" <PAD>", "")
            for question in self.ref_questions.numpy()[:, :self.max_len]]
        self.ref_questions_decoded = [self.ref_questions_decoded[0]] # FOR DEBUGGING.
        self.img_feats = self.clevr_dataset.get_feats_from_img_idx(self.img_idx)  # shape (1024, 14, 14)
        self.state = self.State(torch.LongTensor([self.special_tokens.SOS_idx]).view(1, 1), self.img_feats.unsqueeze(0))
        #self.ref_questions = self.ref_questions[1:2]
        # if self.debug_len_vocab is None:
        #     self.ref_questions = self.clevr_dataset.get_questions_from_img_idx(
        #         img_idx)  # shape (max_len - 1, 10) # used to compute the final reward of the episode.
        #     # self.ref_questions = torch.tensor(self.debug_true_questions)
        # self.ref_questions_decoded = [
        #     self.clevr_dataset.decode(question).replace(" <PAD>", "")
        #     for question in self.ref_questions.numpy()]
        # # self.ref_questions_decoded = ["Are Do"]
        # print(self.ref_questions_decoded)
        self.step_idx = 0
        self.dialog = None
        return self.state

    def get_reduced_action_space(self):
        assert self.ref_questions is not None
        ref_questions = self.ref_questions[:, :self.max_len].reshape(-1)
        ref_questions = ref_questions.data.numpy()
        idx_tokens = np.where(ref_questions!=0)[0]
        ref_questions = list(ref_questions[idx_tokens])
        unique_tokens = list(set(ref_questions))
        reduced_vocab = self.clevr_dataset.idx2word(unique_tokens, delim=',')
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