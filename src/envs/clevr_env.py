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

    def __init__(self, data_path, max_len, reward_type="levenshtein_",
                 reward_path=None, max_samples=None, debug=False, mode="train", num_questions=10, diff_reward=False,
                 condition_answer=True):
        super(ClevrEnv, self).__init__()
        self.mode = mode
        self.data_path = data_path
        modes = {"train": "train", "test_images": "val", "test_text": "train"}
        h5_questions_path = os.path.join(data_path, '{}_questions.h5'.format(modes[self.mode]))
        h5_feats_path = os.path.join(data_path, '{}_features.h5'.format(modes[self.mode]))
        vocab_path = os.path.join(data_path, 'vocab.json')
        self.debug = debug.split(",")
        self.num_questions = num_questions
        self.clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                           h5_feats_path=h5_feats_path,
                                           vocab_path=vocab_path,
                                           max_samples=max_samples)

        SOS_idx = self.clevr_dataset.vocab_questions["<SOS>"]
        EOS_idx = self.clevr_dataset.vocab_questions["<EOS>"]

        Special_Tokens = namedtuple('Special_Tokens', ('SOS_idx', 'EOS_idx'))
        self.special_tokens = Special_Tokens(SOS_idx, EOS_idx)
        self.State = namedtuple('State', ('text', 'img', "answer"))
        self.max_len = max_len

        self.reward_type = reward_type
        self.reward_func = rewards[reward_type](reward_path)
        self.diff_reward = diff_reward
        if diff_reward:
            self.reward_func = Differential(self.reward_func)
        self.step_idx = 0
        self.state, self.dialog = None, None
        self.ref_questions, self.ref_questions_decoded = None, None
        self.img_idx, self.img_feats, self.ref_answer = None, None, None
        self.condition_answer = condition_answer

    def step(self, action):
        # note that the padding of ref questions and generated dialog has been removed.
        action = torch.tensor(action).view(1, 1)
        self.state = self.State(torch.cat([self.state.text, action], dim=1), self.state.img, self.ref_answer)
        done = True if action.item() == self.special_tokens.EOS_idx or self.step_idx == (self.max_len - 1) else False
        question_tokens_padded, question_tokens = np.zeros((self.max_len + 1)), self.state.text.numpy().ravel()
        # question_tokens_padded[:question_tokens.shape[0]] = question_tokens  # if needed
        question = self.clevr_dataset.idx2word(question_tokens, stop_at_end=True)  # remove the EOS token if needed.

        reward, closest_question, pred_answer = self.reward_func.get(question=question,
                                                        ep_questions_decoded=self.ref_questions_decoded,
                                                        step_idx=self.step_idx, done=done, real_answer=self.ref_answer,
                                                        state=self.state)
        self.step_idx += 1
        if done:
            self.dialog = question
        return self.state, (reward, closest_question), done, {}

    def reset(self, seed=None):
        range_images = [int(self.debug[0]), int(self.debug[1])] if self.mode != "test_images" else [0,
                                                                                                    self.clevr_dataset.all_feats.shape[
                                                                                                        0]]
        if seed is not None:
            np.random.seed(seed)
        self.data_idx = np.random.randint(range_images[0], range_images[1]*10)
        #self.img_idx = np.random.randint(range_images[0], range_images[1])
        self.img_idx = self.clevr_dataset.img_idxs[self.data_idx]
        self.ref_questions = self.clevr_dataset.get_questions_from_img_idx(self.img_idx)[:,
                             :self.max_len]  # shape (10, 45)
        if self.mode == "train":
            self.ref_questions = self.ref_questions[0:self.num_questions, :]
        elif self.mode == "test_text":
            self.ref_questions = self.ref_questions[self.num_questions:, :]
        self.ref_questions_decoded = [self.clevr_dataset.idx2word(question, ignored=['<SOS>', '<PAD>'])
                                      for question in self.ref_questions.numpy()]

        _, _, self.ref_answer = self.clevr_dataset[self.data_idx]
        self.img_feats = self.clevr_dataset.get_feats_from_img_idx(self.img_idx)  # shape (1024, 14, 14)

        state_question = [self.special_tokens.SOS_idx]
        # if self.condition_answer:
        # state_question.insert(0, self.clevr_dataset.len_vocab + self.ref_answer)
        self.state = self.State(torch.LongTensor(state_question).view(1, len(state_question)),
                                self.img_feats.unsqueeze(0),self.ref_answer)
        self.step_idx = 0
        self.dialog = None
        # check the correctness of the reward function.
        if self.reward_type == "levenshtein_" and not self.diff_reward:
            reward_true_question, _ = self.reward_func.get(question=self.ref_questions_decoded[0],
                                                           ep_questions_decoded=self.ref_questions_decoded,
                                                           step_idx=self.step_idx, done=True)
            assert reward_true_question == 0, "ERROR IN REWARD FUNCTION"

        return self.state

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
    env = ClevrEnv(data_path="../../data", max_len=20, max_samples=20, debug='0,20', mode="test_images")
    seed = 123
    seed = np.random.randint(100000)
    state = env.reset(seed)
    print('Img Idx', env.img_idx)
    state = env.reset(seed)
    print('Img Idx', env.img_idx)

    make_env_fn = lambda: ClevrEnv(data_path="../../data", max_len=5, max_samples=20)
    # env = VectorEnv(make_env_fn, n=4)
    # env = DummyVecEnv([make_env_fn])
