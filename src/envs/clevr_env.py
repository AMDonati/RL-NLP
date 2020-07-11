import logging
import os
from collections import namedtuple

import gym
import numpy as np
import torch

from RL_toolbox.reward import rewards, Differential
from data_provider.CLEVR_Dataset import CLEVR_Dataset


class ClevrEnv(gym.Env):
    """Clevr Env"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, max_len, reward_type="levenshtein",
                 reward_path=None, max_samples=None, debug=False, mode="train", num_questions=10, diff_reward=False):
        super(ClevrEnv, self).__init__()
        self.mode = mode
        self.data_path = data_path
        h5_questions_path = os.path.join(data_path, '{}_questions.h5'.format(self.mode))
        h5_feats_path = os.path.join(data_path, '{}_features.h5'.format(self.mode))
        vocab_path = os.path.join(data_path, 'vocab.json')
        # self.debug_true_questions = torch.randint(0,debug_len_vocab, (2,))
        self.debug = debug.split(",")
        self.num_questions = num_questions
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
                                  ('img_idx', 'closest_question', 'dialog', 'rewards', 'valid_actions'))
        self.max_len = max_len
        # self.ref_questions = torch.randint(0, self.debug_len_vocab,
        #                                  (3, self.max_len)) if self.debug_len_vocab is not None else None
        # self.reset()

        self.reward_func = rewards[reward_type](reward_path)
        if diff_reward:
            self.reward_func = Differential(self.reward_func)
        self.step_idx = 0
        self.state, self.dialog = None, None
        self.ref_questions, self.ref_questions_decoded = None, None
        self.img_idx, self.img_feats = None, None

    def step(self, action):
        action = torch.tensor(action).view(1, 1)
        self.state = self.State(torch.cat([self.state.text, action], dim=1), self.state.img)
        question_token = np.zeros((self.max_len + 1))
        question_token[:len(self.state.text.numpy()[0])] = self.state.text.numpy()[0]
        question = self.clevr_dataset.idx2word(question_token)
        done = True if action.item() == self.special_tokens.EOS_idx or self.step_idx == (self.max_len - 1) else False
        # question = preprocess_final_state(state_text=self.state.text, dataset=self.clevr_dataset,
        #                               EOS_idx=self.special_tokens.EOS_idx)
        reward, closest_question = self.reward_func.get(question=question,
                                                        ep_questions_decoded=self.ref_questions_decoded,
                                                        step_idx=self.step_idx, done=done)
        self.step_idx += 1
        if done:
            self.dialog = question
            logging.info(question)
        return self.state, (reward, closest_question), done, {}

    def reset(self):
        self.img_idx = np.random.randint(int(self.debug[0]), int(self.debug[1]))
        # self.img_idx = 2
        self.ref_questions = self.clevr_dataset.get_questions_from_img_idx(self.img_idx)[:,
                             :self.max_len]  # shape (10, 45)
        # if self.debug > 0:
        self.ref_questions = self.ref_questions[0:self.num_questions]
        # if self.debug:
        # self.ref_questions = torch.tensor([[7, 8, 10, 12, 14]])
        self.ref_questions_decoded = [self.clevr_dataset.idx2word(question, clean=True)
                                      for question in self.ref_questions.numpy()]
        # logging.info("Questions for image {} : {}".format(self.img_idx, self.ref_questions_decoded))
        # self.ref_questions_decoded = [self.ref_questions_decoded[0]]  # FOR DEBUGGING.
        self.img_feats = self.clevr_dataset.get_feats_from_img_idx(self.img_idx)  # shape (1024, 14, 14)
        self.state = self.State(torch.LongTensor([self.special_tokens.SOS_idx]).view(1, 1), self.img_feats.unsqueeze(0))
        self.step_idx = 0
        self.dialog = None
        self.current_episode = self.Episode(self.img_idx, None, None, None, None)

        return self.state

    def decode_current_episode(self):
        valid_actions = self.current_episode.valid_actions
        assert valid_actions is not None
        valid_actions_decoded = [self.clevr_dataset.idx2word(actions, delim=',') for actions in valid_actions]
        # dialog_split = [self.current_episode.dialog.split()[:i] for i in range(valid_actions)]
        # return dict(zip(dialog_split, valid_actions_decoded))
        return valid_actions_decoded

    def clean_ref_questions(self):
        questions_decoded = [tokens.replace('<PAD>', '') for tokens in self.ref_questions_decoded]
        questions_decoded = [q.strip() for q in questions_decoded]
        self.ref_questions_decoded = questions_decoded

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


class VectorEnv:
    def __init__(self, make_env_fn, n):
        self.envs = tuple(make_env_fn() for _ in range(n))

    # Call this only once at the beginning of training (optional):
    def seed(self, seeds):
        assert len(self.envs) == len(seeds)
        return tuple(env.seed(s) for env, s in zip(self.envs, seeds))

    # Call this only once at the beginning of training:
    def reset(self):
        return tuple(env.reset() for env in self.envs)

    # Call this on every timestep:
    def step(self, actions):
        assert len(self.envs) == len(actions)
        # return_values = []
        obs_batch, rew_batch, done_batch, info_batch = [], [], [], []
        for env, a in zip(self.envs, actions):
            observation, (reward, _), done, info = env.step(a)
            if done:
                observation = env.reset()
            obs_batch.append(observation)
            rew_batch.append(reward)
            done_batch.append(done)
            info_batch.append(info)
            # return_values.append((observation, reward, done, info))
        return obs_batch, rew_batch, done_batch, info_batch
        # return tuple(return_values)

    # Call this at the end of training:
    def close(self):
        for env in self.envs:
            env.close()


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

    make_env_fn = lambda: ClevrEnv(data_path="../../data", max_len=5, max_samples=20)
    # env = VectorEnv(make_env_fn, n=4)
    # env = DummyVecEnv([make_env_fn])
