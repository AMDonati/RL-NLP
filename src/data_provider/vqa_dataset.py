# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import _pickle as cPickle
import json
import logging
import os

import nltk
import pandas as pd
import torch
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
from torch.utils.data import Dataset
import numpy as np
import time

nltk.download('punkt')
from data_provider._image_features_reader import ImageFeaturesH5Reader

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def _create_entry(question, answer):
    answer.pop("image_id")
    answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],
        "question": question["question"],
        "answer": answer,
    }
    return entry


def clean_key(key, tokens_to_remove):
    bool = False
    for tok in tokens_to_remove:
        if tok in key:
            bool = True
    return bool


def clean_dict_keys(dic, tokens_to_remove):
    keys_to_remove = []
    for key in dic.keys():
        for tok in tokens_to_remove:
            if tok in key:
                keys_to_remove.append(key)
    keys_to_remove = list(set(keys_to_remove))
    for entry in keys_to_remove:
        if entry in dic.keys():
            del dic[entry]

    return dic


def _load_dataset(dataroot, name, clean_datasets):
    """Load entries
    dataroot: root path of dataset
    name: 'train', 'val', 'trainval', 'minsval'
    """
    if name == "train" or name == "val":
        question_path = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % name)
        questions = sorted(
            json.load(open(question_path))["questions"], key=lambda x: x["question_id"]
        )
        answer_path = os.path.join(dataroot, "cache", "%s_target.pkl" % name)
        answers = cPickle.load(open(answer_path, "rb"))
        answers = sorted(answers, key=lambda x: x["question_id"])

    elif name == "trainval":
        question_path_train = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "train"
        )
        questions_train = sorted(
            json.load(open(question_path_train))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_train = os.path.join(dataroot, "cache", "%s_target.pkl" % "train")
        answers_train = cPickle.load(open(answer_path_train, "rb"))
        answers_train = sorted(answers_train, key=lambda x: x["question_id"])

        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])
        questions = questions_train + questions_val
        answers = answers_train + answers_val

    elif name == "minval":
        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "val"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["question_id"],
        )
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "val")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["question_id"])
        questions = questions_val[-3000:]
        answers = answers_val[-3000:]

    elif name == "mintrain":
        question_path_val = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2014_questions.json" % "train"
        )
        questions_val = sorted(
            json.load(open(question_path_val))["questions"],
            key=lambda x: x["image_id"],
        )
        answer_path_val = os.path.join(dataroot, "cache", "%s_target.pkl" % "train")
        answers_val = cPickle.load(open(answer_path_val, "rb"))
        answers_val = sorted(answers_val, key=lambda x: x["image_id"])
        questions = questions_val[70000:80000]
        answers = answers_val[70000:80000]

    elif name == "test":
        question_path_test = os.path.join(
            dataroot, "v2_OpenEnded_mscoco_%s2015_questions.json" % "test"
        )
        questions_test = sorted(
            json.load(open(question_path_test))["questions"],
            key=lambda x: x["question_id"],
        )
        questions = questions_test

    else:
        assert False, "data split is not recognized."

    if "test" in name:
        entries = []
        for question in questions:
            entries.append(question)
    else:
        assert_eq(len(questions), len(answers))
        entries = []
        remove_ids = []
        if clean_datasets:
            remove_ids = np.load(os.path.join(dataroot, "cache", "coco_test_ids.npy"))
            remove_ids = [int(x) for x in remove_ids]
        for question, answer in zip(questions, answers):
            if "train" in name and int(question["image_id"]) in remove_ids:
                continue
            assert_eq(question["question_id"], answer["question_id"])
            assert_eq(question["image_id"], answer["image_id"])
            entries.append(_create_entry(question, answer))

    return entries


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
            vocab_path=None,
            tokenize=True,
            max_samples=None,
    rl=True):
        super().__init__()
        self.split = split
        self.get_answers_vocab(dataroot)
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        self._image_features_reader = image_features_reader
        self.question_tokenizer = question_tokenizer
        self.lm_tokenizer = question_tokenizer.lm_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.special_tokens = question_tokenizer.special_tokens
        self._padding_index = question_tokenizer.special_tokens['<PAD>']
        self.max_samples = max_samples
        '''
        '''
        cache_path = os.path.join(
            dataroot,
            "cache",
            split + "_" + "entries.pkl")

        vocab_path_ = os.path.join(
            dataroot,
            "cache",
            "vocab.json")
        if not os.path.isfile(vocab_path_):
            print("Building vocab...")
            self.entries = _load_dataset(dataroot, split, clean_datasets)
            self.build_true_vocab(vocab_path_)
            print("Vocab built...")
        else:
            print("Loading vocab...")
            current_time = time.time()
            self.load_vocab(vocab_path_)
            print("Time for loading vocab:", time.time()- current_time)

        # tokenize with vocab & tensorize
        self.question_tokenizer.set_vocab(self.vocab_questions)
        self.set_traduction_dictionnaries()

        if tokenize:
            if not os.path.exists(cache_path):
                self.entries = _load_dataset(dataroot, split, clean_datasets)
                image_ids = list(map(int, image_features_reader._image_ids[:-1]))
                self.entries = [entry for entry in self.entries if entry["image_id"] in image_ids]
                self.tokenize()
                self.tensorize()
                cPickle.dump(self.entries, open(cache_path, "wb"))
            else:
                logger.info("Loading from %s" % cache_path)
                current_time = time.time()
                self.entries = cPickle.load(open(cache_path, "rb"))
                print("time to load entries:", time.time() - current_time)

            self.len_vocab = len(self.vocab_questions)
            logger.info("vocab size: {}".format(self.len_vocab))
            logger.info("number of answers: {}".format(self.len_vocab_answer))

            # filter entries if needed.
            if filter_entries:
                current_time = time.time()
                self.filter_entries(min_len_questions=min_len_questions, num_answers=num_answers,
                                    filter_yes_no=filter_yes_no,
                                    num_images=num_images)
                print("time to filter entries:", time.time() - current_time)
                if self.split == 'train' and rl:
                    current_time = time.time()
                    self.split_entries()
                    print("time to split entries:", time.time() - current_time)

    def build_true_vocab(self, vocab_out_path, tokens_to_remove=["-", ".", "/", "(", ")", "`", "#", "^", ":"],
                         save_first_words=False):
        true_vocab = self.special_tokens
        first_words = []
        for entry in self.entries:
            tokens = self.lm_tokenizer.encode(entry["question"])
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

    def filter_entries(self, min_len_questions=0, num_answers=1, filter_yes_no=True, num_images=None):
        self.filtered_entries = []
        self.remaining_entries = []
        yes_idx = self.ans2label["yes"]
        no_idx = self.ans2label["no"]
        for entry in self.entries:
            len_q = len(word_tokenize(entry["question"]))
            number_of_answers = len(entry["answer"]["labels"]) if entry["answer"]["labels"] is not None else 0
            if len_q >= min_len_questions and number_of_answers == num_answers:
                if filter_yes_no:
                    if entry["answer"]["labels"][0] != yes_idx and entry["answer"]["labels"][0] != no_idx:
                        self.filtered_entries.append(entry)
                else:
                    self.filtered_entries.append(entry)
            else:
                if entry["answer"]["labels"] is not None:
                    self.remaining_entries.append(entry)
        if num_images is not None:
            df = pd.DataFrame.from_records(self.filtered_entries)
            images_idx = np.sort(df.image_id.unique())
            self.images_idx = images_idx[:num_images]
            self.filtered_entries = [entry for entry in self.filtered_entries if entry["image_id"] in images_idx]
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
                entry["question"])

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

        inputs = question[:-1]
        targets = question[1:]

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
        print("length of image_reader:", len(self._image_features_reader))
        if self.max_samples is not None:
            return min(self.max_samples, len(self._image_features_reader), len(self.filtered_entries))
        else:
            return min(len(self._image_features_reader), len(self.filtered_entries))


if __name__ == '__main__':
    from transformers import BertTokenizer, GPT2Tokenizer
    import argparse
    from data_provider.vqa_tokenizer import VQATokenizer
    import numpy as np

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default='../../data/vqa-v2',
                        help="data folder containing questions embeddings and img features")
    parser.add_argument("-features_path", type=str, default="../../data/vqa-v2/coco_trainval.lmdb",
                        help="data folder containing questions embeddings and img features")
    parser.add_argument("-vocab_path", type=str, default="../../data/vqa-v2/cache/vocab.json")
    parser.add_argument("-split", type=str, default="mintrain")
    parser.add_argument("-test", type=int, default=1)
    args = parser.parse_args()

    lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    reward_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    features_h5path = args.features_path
    images_feature_reader = ImageFeaturesH5Reader(features_h5path, False)
    question_tokenizer = VQATokenizer(lm_tokenizer=lm_tokenizer)

    if args.vocab_path is None:
        vqa_dataset = VQADataset(split="trainval", dataroot=args.data_path,
                                 question_tokenizer=question_tokenizer, image_features_reader=images_feature_reader,
                                 reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                                 num_images=20, tokenize=False)

    else:
        print("Building {} dataset...".format(args.split))
        vqa_dataset = VQADataset(split=args.split, dataroot=args.data_path,
                                 question_tokenizer=question_tokenizer, image_features_reader=images_feature_reader,
                                 reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                                 num_images=20, vocab_path=args.vocab_path)


    if args.test:
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
        print("Test of reward tokenizer...")
        print('Is there a pizza?')
        lm_idx = vqa_dataset.lm_tokenizer.encode('Is there a pizza?')
        input_idx = [vqa_dataset.lm_to_dataset_trad[idx] for idx in lm_idx]
        reward_idx = vqa_dataset.translate_for_reward(input_idx)
        question_decoded = vqa_dataset.reward_tokenizer.decode(reward_idx)
        print('question decoded', question_decoded)

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

