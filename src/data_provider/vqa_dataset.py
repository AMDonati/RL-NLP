# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import _pickle as cPickle
import json
import logging
import os
import nltk
import pandas as pd
import numpy as np
import torch
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
from data_provider.tokenizer import Tokenizer
from data_provider.vqav2_utils import assert_eq, split_question, _load_dataset, clean_key
from collections import Counter
import torch.nn.functional as F

nltk.download('punkt')
from data_provider._image_features_reader import ImageFeaturesH5Reader

logger = logging.getLogger()  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def isfloat(value):
    try:
        float(value)
        return True
    except ValueError:
        return False


def assert_match_split_vocab_path_args(split, vocab_path, vocab_path_min):
    if vocab_path == vocab_path_min:
        assert split == "minval" or split == "mintrain", "You can't used a reduced vocab on train and val split. used minval and mintrain split instead."


class VQADataset(Dataset):
    def __init__(
            self,
            dataroot,
            split,
            image_features_reader,
            question_tokenizer,
            reward_tokenizer,
            clean_datasets,
            max_seq_length=23,
            max_region_num=101,
            filter_entries=True,
            min_len_questions=6,
            num_answers=1,
            filter_yes_no=True,
            num_images=None,
            vocab_path=os.path.join("data/vqa-v2", "cache", "vocab.json"),
            max_samples=None,
            rl=True, filter_numbers=False):
        super().__init__()
        self.split = split
        self.get_answers_vocab(dataroot)
        self.answer_tokenizer = Tokenizer(self.ans2label)
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self.question_tokenizer = question_tokenizer
        self.lm_tokenizer = question_tokenizer.lm_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.special_tokens = question_tokenizer.special_tokens
        self._padding_index = question_tokenizer.special_tokens['<PAD>']
        self.max_samples = max_samples
        self.images_idx = list(map(int, image_features_reader._image_ids[:-1]))
        '''
        '''
        # Building vocab.
        vocab_path_ = os.path.join(
            dataroot,
            "cache",
            "vocab.json")
        vocab_path_min = os.path.join(
            dataroot,
            "cache",
            "vocab_min.json")
        if vocab_path == "none":
            if split == "trainval" and os.path.isfile(vocab_path_):
                print('WARNING: a vocab.json file already exists and will be replaced')
            elif split == "mintrainval" and os.path.isfile(vocab_path_min):
                print('WARNING: a vocab_min.json file already exists and will be replaced')
            print("Building vocab...")
            self.entries = _load_dataset(dataroot, split, clean_datasets)
            vocab_path__ = vocab_path_ if split == "trainval" else vocab_path_min
            self.build_true_vocab(vocab_path__)
            print("Vocab built...")
        else:
            print("Loading vocab...")
            self.load_vocab(vocab_path)

        # tokenize with vocab & tensorize
        self.question_tokenizer.set_vocab(self.vocab_questions)
        self.set_traduction_dictionnaries()

        if vocab_path != "none":
            cache_path = os.path.join(
                dataroot,
                "cache",
                split + "_" + "entries.pkl")
            if vocab_path == vocab_path_min:
                cache_path = os.path.join(
                    dataroot,
                    "cache",
                    split + "_minvocab_" + "entries.pkl")
            if not os.path.exists(cache_path):
                self.entries = _load_dataset(dataroot, split, clean_datasets)
                image_ids = list(map(int, image_features_reader._image_ids[:-1]))
                self.entries = [entry for entry in self.entries if entry["image_id"] in image_ids]
                self.tokenize()
                self.tensorize()
                cPickle.dump(self.entries, open(cache_path, "wb"))
            else:
                print("Loading from %s" % cache_path)
                self.entries = cPickle.load(open(cache_path, "rb"))

            self.len_vocab = len(self.vocab_questions)
            logger.info("vocab size: {}".format(self.len_vocab))
            logger.info("number of answers: {}".format(self.len_vocab_answer))

            # filter entries if needed.
            if filter_entries:
                self.filter_entries(min_len_questions=min_len_questions, num_answers=num_answers,
                                    filter_yes_no=filter_yes_no,
                                    num_images=num_images, filter_floats=filter_numbers)
                if rl:
                    if self.split == 'train' or self.split == 'mintrain':
                        self.split_entries()

                self.reduced_answers = [entry["answer"]["labels"] for entry in self.filtered_entries]
                self.reduced_answers = torch.stack(self.reduced_answers).unique()

    def build_true_vocab(self, vocab_out_path, tokens_to_remove=["-", ".", "/", "(", ")", "`", "#", "^", ":", "?"],
                         save_first_words=False):
        true_vocab = self.special_tokens
        first_words = []
        for entry in self.entries:
            tokens = self.lm_tokenizer.encode(entry["question"], add_prefix_space=True)
            for i, token in enumerate(tokens):
                key = self.lm_tokenizer.decoder[token]
                if key not in true_vocab.keys() and not clean_key(key, tokens_to_remove):
                    true_vocab[key] = token
                if i == 0 and key not in first_words:
                    first_words.append(key)
        vocab = dict(zip(true_vocab.keys(), range(len(true_vocab))))
        print("Number of words in vocab:{}".format(len(vocab)))
        self.vocab_questions = vocab
        first_words_path = vocab_out_path.split(".")[0] + '_first_words.json'
        with open(vocab_out_path, 'w') as f:
            json.dump(self.vocab_questions, f)
        if save_first_words:
            with open(first_words_path, 'w') as f:
                json.dump(first_words, f)

    def set_traduction_dictionnaries(self):
        self.dataset_to_lm_trad = self.question_tokenizer.dataset_to_lm_trad
        self.lm_to_dataset_trad = self.question_tokenizer.lm_to_dataset_trad

    def get_answers_vocab(self, dataroot):
        ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.ans2label = {k: v for k, v in sorted(self.ans2label.items(), key=lambda item: item[1])}
        self.label2ans = {v: k for k, v in self.ans2label.items()}
        self.len_vocab_answer = len(self.ans2label)

    def load_vocab(self, vocab_out_path):
        with open(vocab_out_path, 'r') as f:
            self.vocab_questions = json.load(f)
        print("loading {} words vocab".format(len(self.vocab_questions)))

    def translate_for_reward(self, question_idx):
        question_decoded = self.question_tokenizer.decode(question_idx)
        reward_question_idx = self.reward_tokenizer.encode(question_decoded)
        return reward_question_idx

    def filter_entries(self, min_len_questions=0, num_answers=1, filter_yes_no=True, num_images=None,
                       filter_floats=False):
        self.filtered_entries = []
        self.remaining_entries = []
        yes_idx = self.ans2label["yes"]
        no_idx = self.ans2label["no"]
        # floats = [v for k, v in self.ans2label.items() if isfloat(k)]
        numbers_idx = [self.ans2label[str(i)] for i in range(20) if str(i) in self.ans2label]
        for entry in self.entries:
            len_q = len(word_tokenize(entry["question"]))
            number_of_answers = len(entry["answer"]["labels"]) if entry["answer"]["labels"] is not None else 0
            if len_q >= min_len_questions and number_of_answers == num_answers:
                if filter_floats:
                    if entry["answer"]["labels"][0] in numbers_idx:
                        self.filtered_entries.append(entry)
                elif filter_yes_no:
                    if entry["answer"]["labels"][0] != yes_idx and entry["answer"]["labels"][0] != no_idx:
                        self.filtered_entries.append(entry)
                else:
                    self.filtered_entries.append(entry)
            else:
                if entry["answer"]["labels"] is not None:
                    self.remaining_entries.append(entry)
        if num_images is not None:
            df = pd.DataFrame.from_records(self.filtered_entries)
            images_idx = df.image_id.sort_values().unique()
            self.images_idx = images_idx[:num_images]
            df = df.loc[df['image_id'] <= self.images_idx[-1]]
            self.filtered_entries = df.to_dict(orient="records")
        print("keeping {} entries over {} original entries".format(len(self.filtered_entries), len(self.entries)))
        del self.entries

    def split_entries(self):
        train_entries, test_entries = [], []
        for img_idx in self.images_idx:
            img_entries = [entry for entry in self.filtered_entries if entry["image_id"] == img_idx]
            if len(img_entries) > 1:
                test_entries.append(img_entries[-1])
                img_entries.pop()
            for l in img_entries:
                train_entries.append(l)
        self.filtered_entries = train_entries
        self.test_entries = test_entries
        print("splitting filtered entries between {} for train and {} for test".format(len(self.filtered_entries),
                                                                                       len(self.test_entries)))

    def get_answers_frequency(self):
        answers_idx = [entry["answer"]["labels"].cpu().squeeze().item() for entry in self.filtered_entries]
        freq_answers = Counter(answers_idx)
        inv_freq_norm = F.softmax(torch.tensor([1 / item for item in list(freq_answers.values())], dtype=torch.float32))
        inv_freq_answers = {k: inv_freq_norm[i].item() for i, k in enumerate(list(freq_answers.keys()))}
        return inv_freq_answers

    def get_masks_for_tokens(self, tokens):
        tokens = tokens[: self._max_seq_length - 2]
        segment_ids = [0] * len(tokens)
        input_mask = [1] * len(tokens)

        if len(tokens) < self._max_seq_length:
            # Note here we pad in front of the sentence
            padding = [self._padding_index] * (self._max_seq_length - len(tokens))
            tokens = tokens + padding
            input_mask += padding
            segment_ids += padding
        assert_eq(len(tokens), self._max_seq_length)
        return segment_ids, input_mask, tokens

    def tokenize(self):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens_vil = self.reward_tokenizer.encode(
                entry["question"])
            tokens_lm = self.lm_tokenizer.encode(
                entry["question"], add_prefix_space=True)

            tokens_vil = tokens_vil[: self._max_seq_length - 2]
            segment_ids = [0] * len(tokens_vil)
            input_mask = [1] * len(tokens_vil)

            if len(tokens_vil) < self._max_seq_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (self._max_seq_length - len(tokens_vil))
                tokens_vil = tokens_vil + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens_vil), self._max_seq_length)
            entry["q_token_vilbert"] = tokens_vil
            entry["q_token_lm"] = tokens_lm
            entry["q_token"] = self.tokenize_with_vocab(tokens_lm)
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tokenize_with_vocab(self, tokens_lm):
        tokens = []
        for tok in tokens_lm:
            if tok in self.lm_to_dataset_trad.keys():
                tokens.append(self.lm_to_dataset_trad[tok])
            else:
                if tok in self.special_tokens:
                    tokens.append(tok)

        if len(tokens) < self._max_seq_length:
            padding = [self._padding_index] * (self._max_seq_length - len(tokens))
            tokens = tokens + padding
        elif len(tokens) > self._max_seq_length:
            tokens = tokens[:self._max_seq_length]
        assert_eq(len(tokens), self._max_seq_length)
        return tokens

    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            question_lm = torch.from_numpy(np.array(entry["q_token_lm"]))
            entry["q_token_lm"] = question_lm

            question_vil = torch.from_numpy(np.array(entry["q_token_vilbert"]))
            entry["q_token_vilbert"] = question_vil

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            if "test" not in self.split:
                answer = entry["answer"]
                labels = np.array(answer["labels"])
                scores = np.array(answer["scores"], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None

    def compute_vocab_stats(self):
        df_vqa = pd.DataFrame.from_records(self.entries)
        tok_func = lambda t: [self.question_tokenizer.idx_to_token[i] for i in t.numpy()]
        tokens = df_vqa.q_token.apply(tok_func)
        sum_tokens = tokens.sum()
        tok_dist = FreqDist(sum_tokens)
        return tok_dist

    def get_img_data(self, entry):
        image_id = entry["image_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))

        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)

        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()

        return features, image_mask, spatials

    def get_answer_data(self, entry):
        target = torch.zeros(self.len_vocab_answer)
        if "test" not in self.split:
            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            if labels is not None:
                target.scatter_(0, labels, scores)
        return labels, target

    def __getitem__(self, index, add_sos_token=True):
        entries = self.filtered_entries
        entry = entries[index]

        features, image_mask, spatials = self.get_img_data(entry)
        labels, _ = self.get_answer_data(entry)

        question = entry["q_token"]
        if add_sos_token:
            question = torch.cat([torch.tensor(self.vocab_questions["<SOS>"]).view(1), question])

        inputs, targets = split_question(question)

        return (inputs, targets), labels, (features, image_mask, spatials)

    def get_data_for_ViLBERT(self, index, mode="train"):
        if mode == "test_text":
            entries = self.test_entries
        else:
            entries = self.filtered_entries

        entry = entries[index]

        features, image_mask, spatials = self.get_img_data(entry)
        _, target = self.get_answer_data(entry)
        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        question = entry["q_token_vilbert"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]
        question_id = entry["question_id"]

        return (
            features,
            spatials,
            image_mask,
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id
        )

    def __len__(self):
        if self.max_samples is not None:
            return min(self.max_samples, len(self._image_features_reader), len(self.filtered_entries))
        else:
            return min(len(self._image_features_reader), len(self.filtered_entries))


if __name__ == '__main__':
    from transformers import BertTokenizer, GPT2Tokenizer
    from data_provider.vqa_tokenizer import VQATokenizer
    import numpy as np

    data_path = '../../data/vqa-v2'
    features_path = "../../data/vqa-v2/coco_trainval.lmdb"
    vocab_path = "../../data/vqa-v2/cache/vocab_min.json"

    lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    print("test of lm_tokenizer...")
    ids = lm_tokenizer("The cat is on the mat", add_prefix_space=True)
    print(ids)
    decode_ids = lm_tokenizer.decode(ids["input_ids"])
    print(decode_ids)
    ids_2 = lm_tokenizer.encode("The cat is on the mat", add_prefix_space=True)

    reward_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    features_h5path = features_path
    images_feature_reader = ImageFeaturesH5Reader(features_h5path, False)
    question_tokenizer = VQATokenizer(lm_tokenizer=lm_tokenizer)

    split = "mintrain"
    vqa_dataset = VQADataset(split=split, dataroot=data_path,
                             question_tokenizer=question_tokenizer, image_features_reader=images_feature_reader,
                             reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                             num_images=None, vocab_path=vocab_path)
    test = 1 if split == "mintrain" else 0
    if test:
        vocab = vqa_dataset.vocab_questions
        new_d = {}
        for k in sorted(vocab, key=len):
            new_d[k] = vocab[k]

        # test of answers vocab:
        answers_ids = list(vqa_dataset.ans2label.values())
        print("first id", answers_ids[0])
        print("last id", answers_ids[-1])
        print("len vocab answers", vqa_dataset.len_vocab_answer)

        # test of translate functions:
        # print("Test of reward tokenizer...")
        # print('Is there a pizza?')
        # lm_idx = vqa_dataset.lm_tokenizer.encode('Is there a pizza?')
        # input_idx = [vqa_dataset.lm_to_dataset_trad[idx] for idx in lm_idx]
        # reward_idx = vqa_dataset.translate_for_reward(input_idx)
        # question_decoded = vqa_dataset.reward_tokenizer.decode(reward_idx)
        # print('question decoded', question_decoded)

        print("Test of lm_to_dataset_function ...")
        idx = np.random.randint(vqa_dataset.len_vocab)
        token_idx = list(vqa_dataset.lm_to_dataset_trad.keys())[idx]
        print("word from lm_tokenizer")
        print(vqa_dataset.lm_tokenizer.decoder[token_idx])
        trad_token_idx = vqa_dataset.lm_to_dataset_trad[token_idx]
        print("word from dataset vocab")
        print(vqa_dataset.question_tokenizer.decode([trad_token_idx]))

        print("test of get_item function...")
        (inputs, targets), labels, (features, image_mask, spatials) = vqa_dataset.__getitem__(1)
        print("inputs", inputs)
        print("targets", targets)
        print("answer labels", labels.shape)
        print("features", features.shape)
        print("image_mask", image_mask.shape)
        print("spatials", spatials.shape)

        print("test of get_data_for_VILBERT function...")
        features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = vqa_dataset.get_data_for_ViLBERT(
            index=0)
        print("question", question)
        print("target", target.shape)  # 3129 answers.

        print("print test of decode function...")
        entry = vqa_dataset.filtered_entries[0]
        print("true question:{}".format(entry["question"]))
        print("question decoded - question_tokenizer: {}".format(
            vqa_dataset.question_tokenizer.decode(entry["q_token"].numpy())))
        print("question decoded - lm_tokenizer: {}".format(
            vqa_dataset.lm_tokenizer.decode(entry["q_token_lm"].numpy())))
