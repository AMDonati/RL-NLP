import torch
from torch.utils.data import Dataset
import json
import h5py
import numpy as np
import os

'''
Create a questions Dataset to train the language model. 
Inspired from: https://stanford.edu/~shervine/blog/pytorch-how-to-generate-data-parallel
'''

class QuestionsDataset(Dataset):
  def __init__(self, h5_questions_path, vocab_path):
    super(QuestionsDataset, self).__init__()

    self.data_path = h5_questions_path
    with open(vocab_path, 'r') as f:
      vocab = json.load(f)['question_token_to_idx']
    self.vocab_len = len(vocab)
    inp_questions, _ = self.get_questions()
    self.seq_len = inp_questions.size(-1)

  def get_questions(self):
    hf = h5py.File(self.data_path, 'r')
    input_questions = hf.get('input_questions')
    input_questions = np.array(input_questions)
    input_questions = torch.tensor(input_questions, dtype=torch.int)  # shape (num_samples, seq_len)
    target_questions = hf.get('target_questions')
    target_questions = np.array(target_questions)
    target_questions = torch.tensor(target_questions, dtype=torch.int)
    return input_questions, target_questions

  def __len__(self):
    '''Denotes the total number of samples'''
    inp_questions, _ = self.get_questions()
    return inp_questions.size(0)

  def __getitem__(self, item):
    '''generate one sample of data'''
    input_questions, target_questions = self.get_questions()
    inputs, targets = input_questions[item, :], target_questions[item, :]

    return inputs, targets

if __name__ == '__main__':
  data_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/temp'
  vocab_path = os.path.join(data_path, "vocab_subset_from_train.json")
  train_questions_path = os.path.join(data_path, "train_questions_subset.h5")
  train_dataset = QuestionsDataset(train_questions_path, vocab_path)
  len_train = train_dataset.__len__()
  print('number of training samples', len_train)
  input_0, target_0 = train_dataset.__getitem__(0)
  print('first input sample', input_0)
  print('first input sample', target_0)