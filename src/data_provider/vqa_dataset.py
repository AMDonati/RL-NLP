# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import json
import _pickle as cPickle
import logging

import numpy as np
import torch
from torch.utils.data import Dataset

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
        questions = questions_train + questions_val[:-3000]
        answers = answers_train + answers_val[:-3000]

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
    elif name == "mteval":
        entries = []
        remove_ids = np.load(os.path.join(dataroot, "cache", "coco_test_ids.npy"))
        remove_ids = [int(x) for x in remove_ids]

        for question, answer in zip(questions, answers):
            if int(question["image_id"]) in remove_ids:
                entries.append(_create_entry(question, answer))
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
            task,  # TODO: remove this.
            dataroot,
            split,
            # image_features_reader, #TODO: import image_features_reader from vilbert.
            lm_tokenizer,
            reward_tokenizer,
            clean_datasets,
            special_tokens,
            max_seq_length=16,  # TODO: look at statistics on the dataset.
            max_region_num=101,
    ):
        super().__init__()
        self.split = split
        ans2label_path = os.path.join(dataroot, "cache", "trainval_ans2label.pkl")
        label2ans_path = os.path.join(dataroot, "cache", "trainval_label2ans.pkl")
        self.ans2label = cPickle.load(open(ans2label_path, "rb"))
        self.label2ans = cPickle.load(open(label2ans_path, "rb"))
        self.num_labels = len(self.ans2label)
        self._max_region_num = max_region_num
        self._max_seq_length = max_seq_length
        # self._image_features_reader = image_features_reader
        self.lm_tokenizer = lm_tokenizer
        self.reward_tokenizer = reward_tokenizer
        self.special_tokens = special_tokens
        self._padding_index = special_tokens['<PAD>']
        self.set_tokenizer_special_tokens()
        self.true_vocab = special_tokens

        '''
        Should have: 
        GPT-2 tokenizer
        the "True" vocab: 
        The traduction dictionnary between the GPT-2 vocab & the true vocab. 
        Methods: 
           - idx2word
           - word2idx
        '''

        clean_train = "_cleaned" if clean_datasets else ""
        cache_path = os.path.join(
            dataroot,
            "cache",
            task + "_" + split + "_" + str(max_seq_length) + clean_train + ".pkl")
        vocab_path = os.path.join(
            dataroot,
            "cache",
            task + "_" + split + "_" + str(max_seq_length) + clean_train + "_" "vocab.json")

        if not os.path.exists(cache_path):
            self.entries = _load_dataset(dataroot, split, clean_datasets)
            self.tokenize(max_seq_length)
            self.save_true_vocab(vocab_path)
            self.set_traduction_dictionnaries()
            self.tokenize_with_vocab()
            self.tensorize()
            cPickle.dump(self.entries, open(cache_path, "wb"))
        else:
            logger.info("Loading from %s" % cache_path)
            self.entries = cPickle.load(open(cache_path, "rb"))
            self.load_true_vocab(vocab_path)
            self.set_traduction_dictionnaries()

    def set_tokenizer_special_tokens(self):
        self.lm_tokenizer.eos_token = '<EOS>'
        # self.tokenizer.eos_token_id = self.special_tokens['<EOS>']
        self.lm_tokenizer.bos_token = '<SOS>'
        # self.tokenizer_bos_token_id = self.special_tokens['<SOS>']
        self.lm_tokenizer.pad_token = '<PAD>'
        # self.tokenizer.pad_token_id = self.special_tokens['<PAD>']
        self.lm_tokenizer.unk_token = '<UNK>'
        # self.tokenizer.unk_token_id = self.special_tokens['<UNK>']

    def build_true_vocab(self, tokens):
        for token in tokens:
            # key = self._tokenizer.ids_to_tokens[token]
            key = self.lm_tokenizer.decoder[token]
            if key not in self.true_vocab.keys():
                self.true_vocab[key] = token

    def save_true_vocab(self, vocab_out_path):
        vocab = dict(zip(self.true_vocab.keys(), range(len(self.true_vocab))))  # TODO: add EOS, UNK, SOS, PAD TOKENS.
        self.vocab = vocab
        with open(vocab_out_path, 'w') as f:
            json.dump(self.vocab, f)

    def set_traduction_dictionnaries(self):
        self.dataset_to_lm_trad = {val: self.lm_tokenizer.encoder[key] for key, val in self.vocab.items() if
                                   key in self.lm_tokenizer.encoder.keys()}
        self.lm_to_dataset_trad = {v: k for k, v in self.dataset_to_lm_trad.items()}

    def load_true_vocab(self, vocab_out_path):
        with open(vocab_out_path, 'r') as f:
            self.vocab = json.load(f)

    def idx2word(self, question_idx):
        lm_question_idx = self.translate_for_lm(question_idx)
        question_decoded = self.lm_tokenizer.decode(lm_question_idx)
        return question_decoded

    def word2idx(self, question):
        lm_question_idx = self.lm_tokenizer.encode(question)
        question_idx = [self.lm_to_dataset_trad[idx] for idx in lm_question_idx]
        return question_idx

    def translate_for_lm(self, question_idx):
        lm_question_idx = [self.dataset_to_lm_trad[idx] for idx in question_idx if idx not in self.special_tokens.values()]  # question_idx should not include special tokens.
        return lm_question_idx

    def translate_for_reward(self, question_idx):
        question_decoded = self.idx2word(question_idx)
        reward_question_idx = self.reward_tokenizer.encode(question_decoded)
        return reward_question_idx

    def tokenize(self, max_length=16):
        """Tokenizes the questions.
        This will add q_token in each entry of the dataset.
        -1 represent nil, and should be treated as padding_index in embedding
        """
        for entry in self.entries:
            tokens = self.lm_tokenizer.encode(
                entry["question"])  # TODO: use encode from tokenizer, and then convert to token_idx of the true vocab.
            self.build_true_vocab(tokens)
            tokens = tokens[: max_length - 2]  # TODO: understand this max_length - 2.

            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token_lm"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tokenize_with_vocab(self):
        for entry in self.entries:
            entry["q_token"] = [self.lm_to_dataset_trad[tok] if tok in self.lm_to_dataset_trad.keys() else tok for tok in entry["q_token_lm"]]


    def tensorize(self):
        for entry in self.entries:
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            question_lm = torch.from_numpy(np.array(entry["q_token_lm"])) #TODO: remove padding for
            entry["q_token_lm"] = question_lm

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

    def __getitem__(self, index):
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]

        '''comment img part for now.'''
        # features, num_boxes, boxes, _ = self._image_features_reader[image_id]

        # mix_num_boxes = min(int(num_boxes), self._max_region_num)
        # mix_boxes_pad = np.zeros((self._max_region_num, 5))
        # mix_features_pad = np.zeros((self._max_region_num, 2048))

        # image_mask = [1] * (int(mix_num_boxes))
        # while len(image_mask) < self._max_region_num:
        # image_mask.append(0)

        # mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        # mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]

        # features = torch.tensor(mix_features_pad).float()
        # image_mask = torch.tensor(image_mask).long()
        # spatials = torch.tensor(mix_boxes_pad).float()

        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        segment_ids = entry["q_segment_ids"]

        co_attention_mask = torch.zeros((self._max_region_num, self._max_seq_length))
        target = torch.zeros(self.num_labels)

        if "test" not in self.split:
            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            if labels is not None:
                target.scatter_(0, labels, scores)

        # return (
        #     features,
        #     spatials,
        #     image_mask,
        #     question,
        #     target,
        #     input_mask,
        #     segment_ids,
        #     co_attention_mask,
        #     question_id,
        # )

        return (
            question,
            target,
            input_mask,
            segment_ids,
            co_attention_mask,
            question_id #TODO: add img_id here ?
        )

    def __len__(self):
        return len(self.entries)


if __name__ == '__main__':
    from transformers import BertTokenizer, GPT2Tokenizer
    import argparse

    SPECIAL_TOKENS = {
        '<PAD>': 0,
        '<SOS>': 1,
        '<EOS>': 2,
        '<UNK>': 3,
    }
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default='../../data/vqa-v2',
                        help="data folder containing questions embeddings and img features")
    args = parser.parse_args()
    lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    reward_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    vqa_dataset = VQADataset(task="1_gpt", split="minval", dataroot=args.data_path, lm_tokenizer=lm_tokenizer,
                             reward_tokenizer=reward_tokenizer, special_tokens=SPECIAL_TOKENS, clean_datasets=True)


    # test of translate functions:
    lm_idx = vqa_dataset.lm_tokenizer.encode('Is there a pizza?')
    input_idx = [vqa_dataset.lm_to_dataset_trad[idx] for idx in lm_idx]
    reward_idx = vqa_dataset.translate_for_reward(input_idx)
    question_decoded = vqa_dataset.reward_tokenizer.decode(reward_idx)
    print('question decoded', question_decoded)

    # test of idx2word:
    entry = vqa_dataset.entries[0]
    print('decoded question', vqa_dataset.idx2word(entry['q_token'].numpy()))
    print('question', entry["question"])

    (q, target, input_mask, seqment_id, co_attention_mask, question_id) = vqa_dataset.__getitem__(1)
    print(q)
    print(vqa_dataset.idx2word(q.numpy()))
    print(target.shape) #3129 answers.