'''
Create a questions Dataset to train the language model.
Inspired from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''
import torch
from torch.utils.data import Dataset, DataLoader
import json
import h5py
import numpy as np
import os
from preprocessing.text_functions import decode

#TODO: add a max samples here.
class QuestionsDataset(Dataset):
  def __init__(self, h5_questions_path, vocab_path):
    super(QuestionsDataset, self).__init__()

    self.data_path = h5_questions_path
    self.vocab_path = vocab_path

    self.inp_questions, self.target_questions = self.get_questions()
    self.vocab = self.get_vocab()
    self.idx_to_token = self.get_idx_to_token()

    self.vocab_len = len(self.vocab)
    self.seq_len = self.inp_questions.size(0)

  def get_vocab(self):
    with open(self.vocab_path, 'r') as f:
      vocab = json.load(f)['question_token_to_idx']
    return vocab

  def get_idx_to_token(self):
    idx_to_token = dict(zip(list(self.vocab.values()), list(self.vocab.keys())))
    return idx_to_token

  def idx2word(self, seq_idx, delim=' ', stop_at_end=False):
    tokens = decode(seq_idx=seq_idx, idx_to_token=self.idx_to_token, stop_at_end=stop_at_end, delim=delim)
    return tokens

  def get_questions(self):
    hf = h5py.File(self.data_path, 'r')
    input_questions = hf.get('input_questions')
    input_questions = np.array(input_questions)
    input_questions = torch.tensor(input_questions, dtype=torch.int)  # shape (num_samples, seq_len)
    input_questions = input_questions.t()
    target_questions = hf.get('target_questions')
    target_questions = np.array(target_questions)
    target_questions = torch.tensor(target_questions, dtype=torch.int)
    target_questions = target_questions.t()
    return input_questions, target_questions # dim (S,B)

  def __len__(self):
    '''Denotes the total number of samples'''
    return self.inp_questions.size(1)

  def __getitem__(self, item):
    '''generate one sample of data'''
    inputs, targets = self.inp_questions[:, item], self.target_questions[:, item]
    return inputs, targets

if __name__ == '__main__':
  data_path = '../../data/CLEVR_v1.0/temp'
  vocab_path = os.path.join(data_path, "vocab_subset_from_train.json")
  train_questions_path = os.path.join(data_path, "train_questions.h5")

  train_dataset = QuestionsDataset(train_questions_path, vocab_path)

  len_train = train_dataset.__len__()
  print('number of training samples', len_train)
  input_0, target_0 = train_dataset.__getitem__(0)
  print('first input sample', input_0)
  print('first input sample', target_0)
  train_generator = DataLoader(dataset=train_dataset, batch_size=64)
  for batch, (inp, tar) in enumerate(train_generator):
    if batch == 0:
      print('input', inp.shape)
      print('target', tar.shape)
  vocab = train_dataset.vocab
  idx_to_token = train_dataset.idx_to_token
  assert len(vocab) == len(idx_to_token)

