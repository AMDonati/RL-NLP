import random
from collections import namedtuple

import gym
from pytorch_transformers import BertTokenizer
from transformers import GPT2Tokenizer

from RL_toolbox.reward import rewards, Differential
from data_provider.CLEVR_Dataset import CLEVR_Dataset
from data_provider._image_features_reader import ImageFeaturesH5Reader
from data_provider.vqa_dataset import VQADataset
from data_provider.vqa_tokenizer import VQATokenizer
import numpy as np
import torch
import os


class GenericEnv(gym.Env):
    """Generic Env"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, max_len, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                 reward_type="levenshtein", reward_path=None, mode="train", diff_reward=False, debug=False,
                 condition_answer=True, reward_vocab=None, mask_answers=False, reduced_answers=False):
        super(GenericEnv, self).__init__()
        self.mode = mode
        self.data_path = data_path
        self.mask_answers = mask_answers
        self.max_len = max_len
        self.device = device
        self.condition_answer = condition_answer
        self.debug = debug.split(",") if debug is not None else debug

        self.State = namedtuple('State', ('text', 'img', "answer"))
        # init env.
        self.step_idx = 0
        self.state, self.dialog = None, None
        self.ref_questions, self.ref_questions_decoded = None, None
        self.img_idx, self.img_feats, self.ref_answer = None, None, None
        self.reduced_answers = reduced_answers
        self.min_data = 0

    def set_special_tokens(self):
        SOS_idx = self.dataset.vocab_questions["<SOS>"]
        EOS_idx = self.dataset.vocab_questions["<EOS>"]
        question_mark_idx = self.dataset.vocab_questions["?"]
        Special_Tokens = namedtuple('Special_Tokens', ('SOS_idx', 'EOS_idx', "question_mark_idx"))
        self.special_tokens = Special_Tokens(SOS_idx, EOS_idx, question_mark_idx)

    def set_reward_function(self, reward_type, reward_path, reward_vocab, diff_reward):
        self.reward_type = reward_type
        self.reward_func = rewards[reward_type](path=reward_path, vocab=reward_vocab, dataset=self.dataset, env=self)
        self.diff_reward = diff_reward
        if diff_reward:
            self.reward_func = Differential(self.reward_func)

    def step(self, action):
        action = torch.tensor(action).view(1, 1)
        self.state = self.State(torch.cat([self.state.text, action], dim=1), self.state.img, self.ref_answer)
        done = self.check_if_done(action)
        question_tokens = self.state.text.numpy().ravel()
        question = self.dataset.question_tokenizer.decode(question_tokens)  # remove the EOS token if needed.
        reward, closest_question, pred_answer = self.reward_func.get(question=question,
                                                                     ep_questions_decoded=self.ref_questions_decoded,
                                                                     step_idx=self.step_idx, done=done,
                                                                     real_answer=self.ref_answer,
                                                                     state=self.state)
        self.step_idx += 1
        return self.state, (reward, closest_question, pred_answer), done, {}

    def check_if_done(self, action):
        done = False
        is_action_terminal = action.item() in [self.special_tokens.EOS_idx, self.special_tokens.question_mark_idx]
        if is_action_terminal or self.step_idx == (self.max_len - 1):
            done = True
        return done

    def reset(self, seed):
        pass

    def render(self, mode='human', close=False):
        pass


class ClevrEnv(GenericEnv):
    """Clevr Env"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, max_len, reward_type="levenshtein",
                 reward_path=None, max_samples=None, debug=None, mode="train", num_questions=10, diff_reward=False,
                 condition_answer=True, reward_vocab=None, mask_answers=False,
                 device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), reduced_answers=False):
        super(ClevrEnv, self).__init__(data_path, max_len, reward_type=reward_type,
                                       reward_path=reward_path, mode=mode, debug=debug, diff_reward=diff_reward,
                                       condition_answer=condition_answer, reward_vocab=reward_vocab, mask_answers=False,
                                       device=device, reduced_answers=reduced_answers)

        modes = {"train": "train", "test_images": "val", "test_text": "train"}
        h5_questions_path = os.path.join(data_path, '{}_questions.h5'.format(modes[self.mode]))
        h5_feats_path = os.path.join(data_path, '{}_features.h5'.format(modes[self.mode]))
        vocab_path = os.path.join(data_path, 'vocab.json')
        self.dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                     h5_feats_path=h5_feats_path,
                                     vocab_path=vocab_path,
                                     max_samples=max_samples, mask_answers=mask_answers)

        self.num_questions = num_questions
        self.set_special_tokens()
        self.set_reward_function(reward_type=reward_type, reward_path=reward_path, reward_vocab=reward_vocab,
                                 diff_reward=diff_reward)

    def reset(self, seed=None):
        range_images = [int(self.debug[0]), int(self.debug[1])] if self.mode != "test_images" else [0,
                                                                                                    self.dataset.all_feats.shape[
                                                                                                        0]]
        if seed is not None:
            np.random.seed(seed)
        # getting the environment's elements: Img, ref_questions, ref_answers.
        self.img_idx = np.random.randint(range_images[0], range_images[1])
        self.img_feats, questions, self.ref_answers = self.dataset.get_data_from_img_idx(self.img_idx)
        self.ref_questions = questions[:, :self.max_len]

        # differentiating between the environment modes.
        if self.mode == "train" and not self.mask_answers:
            self.ref_questions = self.ref_questions[0:self.num_questions, :]
            self.ref_answers = self.ref_answers[0:self.num_questions]
        elif self.mode == "test_text" and not self.mask_answers:
            self.ref_questions = self.ref_questions[self.num_questions:, :]
            self.ref_answers = self.ref_answers[self.num_questions:]

        # getting the ref_idx for the couple (question, answer).
        self.ref_question_idx = random.choice(range(self.ref_questions.size(0)))
        self.ref_question = self.ref_questions[self.ref_question_idx]
        self.ref_answer = self.ref_answers[self.ref_question_idx]

        if self.condition_answer != "none":
            self.ref_questions = self.ref_questions[
                                 self.ref_question_idx:self.ref_question_idx + 1]
            self.ref_answers = self.ref_answers[self.ref_question_idx:self.ref_question_idx + 1]

        self.ref_questions_decoded = [self.dataset.question_tokenizer.decode(question, ignored=['<SOS>', '<PAD>'])
                                      for question in self.ref_questions.numpy()]

        # initializing the state.
        state_question = [self.special_tokens.SOS_idx]
        self.state = self.State(torch.LongTensor(state_question).view(1, len(state_question)),
                                self.img_feats.unsqueeze(0), self.ref_answer)
        self.step_idx = 0
        self.dialog = None

        # check the correctness of the reward function.
        if self.reward_type == "levenshtein" and not self.diff_reward:
            reward_true_question, _, _ = self.reward_func.get(question=self.ref_questions_decoded[0],
                                                              ep_questions_decoded=self.ref_questions_decoded,
                                                              step_idx=self.step_idx, done=True)
            assert reward_true_question == 0, "ERROR IN REWARD FUNCTION"

        return self.state


class VQAEnv(GenericEnv):
    """VQA Env"""
    metadata = {'render.modes': ['human']}

    def __init__(self, data_path, features_h5path, max_len=10,
                 reward_type="levenshtein",
                 debug=None,
                 reward_path=None, mode="train", diff_reward=False,
                 condition_answer=True, reward_vocab=None, mask_answers=False, max_seq_length=23, min_len_questions=0,
                 num_answers=1, device=torch.device("cuda" if torch.cuda.is_available() else "cpu"), min_data=0,
                 reduced_answers=False):
        super(VQAEnv, self).__init__(data_path, max_len, reward_type=reward_type,
                                     reward_path=reward_path, debug=debug, mode=mode, diff_reward=diff_reward,
                                     condition_answer=condition_answer, reward_vocab=reward_vocab,
                                     mask_answers=mask_answers, device=device, reduced_answers=reduced_answers)

        # Loading VQA Dataset.
        num_images = int(self.debug[1]) if self.debug is not None else self.debug
        if self.mode == "test_images":
            num_images = None
        lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        question_tokenizer = VQATokenizer(lm_tokenizer=lm_tokenizer)
        reward_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
        images_feature_reader = ImageFeaturesH5Reader(features_h5path, False)
        modes = self.get_modes(device=device, min_data=min_data)
        self.min_data = min_data
        vocab_path = os.path.join(data_path, "cache", "vocab.json") if not min_data else os.path.join(data_path,
                                                                                                      "cache",
                                                                                                      "vocab_min.json")

        self.dataset = VQADataset(split=modes[self.mode], dataroot=data_path,
                                  image_features_reader=images_feature_reader, question_tokenizer=question_tokenizer,
                                  reward_tokenizer=reward_tokenizer, clean_datasets=True,
                                  max_seq_length=max_seq_length, min_len_questions=min_len_questions,
                                  num_answers=num_answers, num_images=num_images, filter_entries=True,
                                  vocab_path=vocab_path)
        self.set_special_tokens()
        self.set_reward_function(reward_type=reward_type, reward_path=reward_path, reward_vocab=reward_vocab,
                                 diff_reward=diff_reward)

    def get_modes(self, device, min_data):
        if min_data or device.type == "cpu":
            modes = {"train": "mintrain", "test_images": "minval", "test_text": "mintrain"}
        else:
            modes = {"train": "train", "test_images": "val", "test_text": "train"}
        return modes

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        entries = self.dataset.test_entries if self.mode == "test_text" else self.dataset.filtered_entries
        self.env_idx = np.random.randint(0, len(entries))
        self.entry = entries[self.env_idx]
        (features, image_mask, spatials) = self.dataset.get_img_data(self.entry)
        labels, _ = self.dataset.get_answer_data(self.entry)
        self.ref_question_idx = self.entry["question_id"]
        self.ref_question = self.entry["q_token"][:self.max_len]
        self.ref_questions = self.ref_question.view(1, -1)
        self.ref_question_decoded = self.dataset.question_tokenizer.decode(self.entry["q_token"][:self.max_len].numpy())
        # self.ref_question_decoded = " ".join(self.ref_question_decoded)
        self.ref_questions_decoded = [self.ref_question_decoded]
        self.ref_answer = labels
        self.img_idx = self.entry["image_id"]
        self.img_feats = features
        self.img = (features, image_mask, spatials)

        # initializing the state.
        state_question = [self.special_tokens.SOS_idx]
        self.state = self.State(torch.LongTensor(state_question).view(1, len(state_question)),
                                self.img_feats.unsqueeze(0), self.ref_answer)
        self.step_idx = 0
        self.dialog = None

        return self.state


if __name__ == '__main__':
    print("Testing Clevr Env...")
    env = ClevrEnv(data_path="../../data", max_len=20, max_samples=20, debug='0,20', mode="test_images")
    seed = 123
    seed = np.random.randint(100000)
    state_clevr = env.reset(seed)
    print('Img Idx', env.img_idx)  # scalar
    print('Question Idx', env.ref_question_idx)
    print('Ref questions', env.ref_questions_decoded)
    print('Ref Answers', env.ref_answers)  # shape (1)
    print('Ref Answer', env.ref_answer)  # scalar
    print('Ref Question', env.ref_question)  # shape (S)
    print("State - text part", state_clevr.text)  # shape (1,S)

    print("Testing VQA Env...")
    vqa_data_path = '../../data/vqa-v2'
    env_vqa = VQAEnv(data_path=vqa_data_path, features_h5path="../../data/vqa-v2/coco_trainval.lmdb", mode="mintrain",
                     max_seq_length=16, debug="0,20")
    env_vqa.mode = "test_text"
    env_vqa.reset()
    env_vqa.mode = "train"

    print("Testing seed for VQA env...")
    seed = 1
    env_vqa.reset(seed=seed)
    print("seed {}".format(seed), "env_idx:{}".format(env_vqa.env_idx), env_vqa.ref_question_decoded, "answer:",
          env_vqa.ref_answer)
    env_vqa.reset(seed=seed)
    print("seed {}".format(seed), "env_idx:{}".format(env_vqa.env_idx), env_vqa.ref_question_decoded, "answer:",
          env_vqa.ref_answer)
    seed = 2
    env_vqa.reset(seed=seed)
    print("seed {}".format(seed), "env_idx:{}".format(env_vqa.env_idx), env_vqa.ref_question_decoded, "answer:",
          env_vqa.ref_answer)
    env_vqa.reset(seed=seed)
    print("seed {}".format(seed), "env_idx:{}".format(env_vqa.env_idx), env_vqa.ref_question_decoded, "answer:",
          env_vqa.ref_answer)

    print("---------------------------------------------------------")
    print(len(env.dataset.vocab_questions))
    state = env_vqa.reset()
    print("State idx", env_vqa.env_idx)
    print('Img Idx', env_vqa.img_idx)
    print('Question Idx', env_vqa.ref_question_idx)
    print('Ref question', env_vqa.ref_question)
    print("Ref Question decoded", env_vqa.ref_question_decoded)
    print('Ref Answer', env_vqa.ref_answer)
    print("entry", env_vqa.entry)

    print("checking step function for VQA env...")
    state, (reward, closest_question, pred_answer), done, _ = env_vqa.step(np.array(6))
    state, (reward, closest_question, pred_answer), done, _ = env_vqa.step(np.array(8))
    state, (reward, closest_question, pred_answer), done, _ = env_vqa.step(np.array(9))
    state, (reward, closest_question, pred_answer), done, _ = env_vqa.step(np.array(7))
    print("state - text part", state.text)  # shape (1,S).
    print("state decoded", env_vqa.dataset.question_tokenizer.decode(state.text.numpy().ravel()))
    print("reward", reward)
    print("closest_question", closest_question)
    print("pred answer", pred_answer)
