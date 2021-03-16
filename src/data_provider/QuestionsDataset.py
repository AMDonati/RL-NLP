'''
Create a questions Dataset to train the language model.
Inspired from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''
import json
import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from preprocessing.text_functions import decode
from data_provider.tokenizer import Tokenizer
import torch.nn.functional as F


# TODO: add a max samples here: select 350,000 questions.
class QuestionsDataset(Dataset):
    def __init__(self, h5_questions_path, vocab_path, range_samples=None):
        super(QuestionsDataset, self).__init__()
        self.data_path = h5_questions_path
        self.vocab_path = vocab_path
        self.range_samples = range_samples
        self.inp_questions, self.target_questions = self.get_questions()
        self.vocab_questions = self.get_vocab()
        self.idx_to_token = self.get_idx_to_token()
        self.len_vocab = len(self.vocab_questions)
        self.seq_len = self.inp_questions.size(1)
        self.question_tokenizer = Tokenizer(self.vocab_questions)

    def get_vocab(self):
        with open(self.vocab_path, 'r') as f:
            vocab = json.load(f)['question_token_to_idx']
        return vocab

    def get_idx_to_token(self):
        idx_to_token = dict(zip(list(self.vocab_questions.values()), list(self.vocab_questions.keys())))
        return idx_to_token

    def idx2word(self, seq_idx, delim=' ', stop_at_end=False):
        tokens = decode(seq_idx=seq_idx, idx_to_token=self.idx_to_token, stop_at_end=stop_at_end, delim=delim)
        return tokens

    def get_questions(self):
        target_questions = []
        input_questions = []
        if os.path.isdir(self.data_path):
            print("getting clevr ext")
            for file in os.listdir(self.data_path):
                if file.endswith(".h5"):
                    questions_hf = h5py.File(os.path.join(self.data_path, file), 'r')
                    input_questions_ext = questions_hf.get('input_questions')
                    input_questions_ext = np.pad(input_questions_ext,
                                                 ((0, 0), (0, 46 - input_questions_ext.shape[1])),
                                                 "constant")
                    input_questions.append(input_questions_ext)

                    target_questions_ext = questions_hf.get('target_questions')
                    target_questions_ext = np.pad(target_questions_ext,
                                                  ((0, 0), (0, 46 - target_questions_ext.shape[1])),
                                                  "constant")
                    target_questions.append(target_questions_ext)


        input_questions = np.concatenate(input_questions)

        target_questions = np.concatenate(target_questions)

        input_questions = torch.LongTensor(input_questions)  # shape (num_samples, seq_len)
        range_samples = list(map(int, self.range_samples.split(" "))) if self.range_samples is not None else [0,
                                                                                                              input_questions.size(
                                                                                                                  0)]
        input_questions = input_questions[range_samples[0]:range_samples[1]]

        target_questions = torch.LongTensor(target_questions)
        target_questions = target_questions[range_samples[0]:range_samples[1]]
        return input_questions, target_questions  # dim (B,S)

    def __len__(self):
        '''Denotes the total number of samples'''
        return self.inp_questions.size(0)

    def __getitem__(self, item):
        '''generate one sample of data'''
        inputs, targets = self.inp_questions[item, :], self.target_questions[item, :]
        return inputs, targets


if __name__ == '__main__':
    data_path = '../../data/CLEVR_v1.0/temp/5000_2000_samples'
    vocab_path = os.path.join(data_path, "vocab_subset_from_train.json")
    train_questions_path = os.path.join(data_path, "train_questions.h5")

    train_dataset = QuestionsDataset(train_questions_path, vocab_path)

    len_train = train_dataset.__len__()
    print('number of training samples', len_train)
    input_0, target_0 = train_dataset.__getitem__([0, 1])
    print('first input sample', input_0)
    print('first input sample', target_0)
    train_generator = DataLoader(dataset=train_dataset, batch_size=64)
    for batch, (inp, tar) in enumerate(train_generator):
        if batch == 0:
            print('input', inp.shape)
            print('target', tar.shape)
    vocab = train_dataset.vocab_questions
    idx_to_token = train_dataset.idx_to_token
    assert len(vocab) == len(idx_to_token)
