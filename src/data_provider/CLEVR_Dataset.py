# https://github.com/facebookresearch/clevr-iep/blob/master/iep/data.py
# collate_fn in image captioning tuto: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/datasets.py
import torch
from torch.utils.data import Dataset, DataLoader
import json
import h5py
import numpy as np
import os

#TODO: add idx_to_token function.

class CLEVR_Dataset(Dataset):
  def __init__(self, h5_questions_path, h5_feats_path, vocab_path, max_samples=None):
    self.questions_path = h5_questions_path
    self.features_path = h5_feats_path
    self.vocab_path = vocab_path
    self.vocab_questions = self.get_vocab('question_token_to_idx')
    self.vocab_answers = self.get_vocab('answer_token_to_idx')
    self.len_vocab = len(self.vocab_questions)
    self.max_samples = max_samples

    # load dataset objects in memory except img:
    questions_hf = h5py.File(self.questions_path, 'r')
    self.input_questions = self.load_data_from_h5(questions_hf.get('input_questions')).t()
    self.target_questions = self.load_data_from_h5(questions_hf.get('target_questions')).t()
    self.img_idxs = self.load_data_from_h5(questions_hf.get('img_idxs'))
    self.answers = self.load_data_from_h5(questions_hf.get('answers'))
    self.feats_shape = self.get_feats_from_img_idx(0).shape

  def get_vocab(self, key):
    with open(self.vocab_path, 'r') as f:
      vocab = json.load(f)[key]
    return vocab

  def load_data_from_h5(self, dataset):
    arr = np.array(dataset, dtype=np.int64)
    tensor = torch.LongTensor(arr)
    return tensor

  def get_feats_from_img_idx(self, img_idx):
    feats_hf = h5py.File(self.features_path, 'r')
    feats = feats_hf.get('features')[img_idx]
    feats = torch.FloatTensor(np.array(feats, dtype=np.float32))
    return feats

  def get_questions_from_img_idx(self, img_idx):
    img_idxs = self.img_idxs
    select_idx = list(np.where(img_idxs.data.numpy() == img_idx))
    select_questions = self.input_questions[:, select_idx]
    return select_questions[1:, :].squeeze(1)  # removing <sos> token.

  def __getitem__(self, index):
    input_question = self.input_questions[:, index]
    target_question = self.target_questions[:, index]
    img_idx = self.img_idxs[index]
    answer = self.answers[index]
    # loading img feature of img_idx
    feats = self.get_feats_from_img_idx(img_idx)

    return (input_question, target_question), feats, answer

  def __len__(self):
    if self.max_samples is None:
      return self.input_questions.size(1)
    else:
      return min(self.max_samples, self.input_questions.size(1))

if __name__ == '__main__':
  data_path = '../../data'
  vocab_path = os.path.join(data_path, "vocab.json")
  h5_questions_path = os.path.join(data_path, "train_questions.h5")
  h5_feats_path = os.path.join(data_path, "train_features.h5") # Caution, here train_features.h5 corresponds only to the first 21 img of the train dataset.
  clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                h5_feats_path=h5_feats_path,
                                vocab_path=vocab_path)
  num_samples = clevr_dataset.__len__()
  print('length dataset', num_samples)
  index = np.random.randint(0, num_samples)
  (inp_q, tar_q), feats, answer = clevr_dataset.__getitem__(0)
  print('inp_q', inp_q.shape)
  print('tar_q', tar_q.shape)
  print('feats', feats.shape)
  print('answer', answer)
  img_idxs = clevr_dataset.img_idxs
  print('subset of images idx', img_idxs[:50].numpy())
  temp_idxs = img_idxs[:200].data.numpy()
  indices_0 = list(np.where(temp_idxs == 0))
  print('indices of img id 0', indices_0)
  questions_img_0 = clevr_dataset.input_questions[:, indices_0]
  print('questions from img 0', questions_img_0.shape)
  #TODO: add Customized DataLoader with customized collate_fn.