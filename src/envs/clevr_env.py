import os
import random
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
                 condition_answer=True, reward_vocab=None, mask_answers=False, path_images="CLEVR_v1.0"):
        super(ClevrEnv, self).__init__()
        self.mode = mode
        self.data_path = data_path
        modes = {"train": "train", "test_images": "val", "test_text": "train"}
        self.clevr_mode=modes[self.mode]
        h5_questions_path = os.path.join(data_path, '{}_questions.h5'.format(modes[self.mode]))
        h5_feats_path = os.path.join(data_path, '{}_features.h5'.format(modes[self.mode]))
        vocab_path = os.path.join(data_path, 'vocab.json')
        self.debug = debug.split(",")
        self.num_questions = num_questions
        self.mask_answers = mask_answers
        self.clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                           h5_feats_path=h5_feats_path,
                                           vocab_path=vocab_path,
                                           max_samples=max_samples, mask_answers=mask_answers)

        SOS_idx = self.clevr_dataset.vocab_questions["<SOS>"]
        EOS_idx = self.clevr_dataset.vocab_questions["<EOS>"]
        question_mark_idx = self.clevr_dataset.vocab_questions["?"]

        Special_Tokens = namedtuple('Special_Tokens', ('SOS_idx', 'EOS_idx', "question_mark_idx"))
        self.special_tokens = Special_Tokens(SOS_idx, EOS_idx, question_mark_idx)
        self.State = namedtuple('State', ('text', 'img', "answer"))
        self.max_len = max_len

        self.reward_type = reward_type
        self.reward_func = rewards[reward_type](path=reward_path, vocab=reward_vocab, dataset=self.clevr_dataset)
        self.diff_reward = diff_reward
        if diff_reward:
            self.reward_func = Differential(self.reward_func)
        self.step_idx = 0
        self.state, self.dialog = None, None
        self.ref_questions, self.ref_questions_decoded = None, None
        self.img_idx, self.img_feats, self.ref_answer = None, None, None
        self.condition_answer = condition_answer
        self.path_images = path_images

    def check_if_done(self, action):
        done = False
        is_action_terminal = action.item() in [self.special_tokens.EOS_idx]
        if is_action_terminal or self.step_idx == (self.max_len - 1):
            done = True
        return done

    def step(self, action):
        # note that the padding of ref questions and generated dialog has been removed.
        action = torch.tensor(action).view(1, 1)
        self.state = self.State(torch.cat([self.state.text, action], dim=1), self.state.img, self.ref_answer)
        done = self.check_if_done(action)
        question_tokens_padded, question_tokens = np.zeros((self.max_len + 1)), self.state.text.numpy().ravel()
        # question_tokens_padded[:question_tokens.shape[0]] = question_tokens  # if needed
        question = self.clevr_dataset.idx2word(question_tokens, stop_at_end=True)  # remove the EOS token if needed.

        reward, closest_question, pred_answer = self.reward_func.get(question=question,
                                                                     ep_questions_decoded=self.ref_questions_decoded,
                                                                     step_idx=self.step_idx, done=done,
                                                                     real_answer=self.ref_answer,
                                                                     state=self.state)
        self.step_idx += 1
        if done:
            self.dialog = question
        return self.state, (reward, closest_question, pred_answer), done, {}

    def reset(self, seed=None):
        range_images = [int(self.debug[0]), int(self.debug[1])] if self.mode != "test_images" else [0,
                                                                                                    self.clevr_dataset.all_feats.shape[
                                                                                                        0]]
        if seed is not None:
            np.random.seed(seed)
        self.img_idx = np.random.randint(range_images[0], range_images[1])
        self.img_feats, questions, self.ref_answers = self.clevr_dataset.get_data_from_img_idx(self.img_idx)
        self.ref_questions = questions[:, :self.max_len]

        if self.mode == "train" and not self.mask_answers:
            self.ref_questions = self.ref_questions[0:self.num_questions, :]
            self.ref_answers = self.ref_answers[0:self.num_questions]
        elif self.mode == "test_text" and not self.mask_answers:
            self.ref_questions = self.ref_questions[self.num_questions:, :]
            self.ref_answers = self.ref_answers[self.num_questions:]

        self.ref_question_idx = random.choice(range(self.ref_questions.size(0)))

        self.ref_question = self.ref_questions[self.ref_question_idx]
        self.ref_answer = self.ref_answers[self.ref_question_idx]
        if self.condition_answer != "none":
            self.ref_questions = self.ref_questions[self.ref_question_idx:self.ref_question_idx + 1]
            self.ref_answers = self.ref_answers[self.ref_question_idx:self.ref_question_idx + 1]

        self.ref_questions_decoded = [self.clevr_dataset.idx2word(question, ignored=['<SOS>', '<PAD>'])
                                      for question in self.ref_questions.numpy()]

        state_question = [self.special_tokens.SOS_idx]
        self.state = self.State(torch.LongTensor(state_question).view(1, len(state_question)),
                                self.img_feats.unsqueeze(0), self.ref_answer)
        self.step_idx = 0
        self.dialog = None
        # check the correctness of the reward function.
        if self.reward_type == "levenshtein_" and not self.diff_reward:
            reward_true_question, _, _ = self.reward_func.get(question=self.ref_questions_decoded[0],
                                                              ep_questions_decoded=self.ref_questions_decoded,
                                                              step_idx=self.step_idx, done=True)
            assert reward_true_question == 0, "ERROR IN REWARD FUNCTION"

        return self.state

    def render(self, mode='human', close=False):
        pass


if __name__ == '__main__':
    env = ClevrEnv(data_path="../../data", max_len=20, max_samples=20, debug='0,20', mode="test_images")
    seed = 123
    seed = np.random.randint(100000)
    state = env.reset(seed)
    print('Img Idx', env.img_idx)
    print('Question Idx', env.ref_question_idx)
    print('Ref questions', env.ref_questions_decoded)
    print('Ref Answers', env.ref_answers)
    print('Ref Answer', env.ref_answer)
    print('Ref Question', env.ref_question)
