# https://github.com/facebookresearch/clevr-iep/blob/master/iep/data.py
# collate_fn in image captioning tuto: https://github.com/yunjey/pytorch-tutorial/blob/master/tutorials/03-advanced/image_captioning/data_loader.py
# https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning/blob/master/datasets.py
import json
import os

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from preprocessing.text_functions import decode, encode


class CLEVR_Dataset(Dataset):
    def __init__(self, h5_questions_path, h5_feats_path, vocab_path, max_samples=None, debug_len_vocab=None):
        self.questions_path = h5_questions_path
        self.features_path = h5_feats_path
        self.vocab_path = vocab_path
        self.debug_len_vocab = debug_len_vocab
        self.vocab_questions = self.get_vocab('question_token_to_idx')
        self.vocab_answers = self.get_vocab('answer_token_to_idx')
        self.len_vocab = len(self.vocab_questions)
        self.idx_to_token = self.get_idx_to_token()
        self.max_samples = max_samples

        # load feats in memory.
        feats_hf = h5py.File(self.features_path, 'r')
        self.all_feats = feats_hf.get('features')  # TODO: eventually add max pool here.

        # load dataset objects in memory except img:
        questions_hf = h5py.File(self.questions_path, 'r')
        self.input_questions = self.load_data_from_h5(questions_hf.get('input_questions'))
        self.target_questions = self.load_data_from_h5(questions_hf.get('target_questions'))
        self.img_idxs = self.load_data_from_h5(questions_hf.get('img_idxs'))
        self.answers = self.load_data_from_h5(questions_hf.get('answers'))
        self.feats_shape = self.get_feats_from_img_idx(0).shape

    def get_vocab(self, key):
        with open(self.vocab_path, 'r') as f:
            vocab = json.load(f)[key]
        return vocab

    def load_data_from_h5(self, dataset):
        arr = np.array(dataset, dtype=np.int32)
        tensor = torch.LongTensor(arr)
        return tensor

    def get_feats_from_img_idx(self, img_idx):
        feats = self.all_feats[img_idx]
        feats = torch.FloatTensor(np.array(feats, dtype=np.float32))
        return feats

    def get_questions_from_img_idx(self, img_idx):
        # caution: this works only for a single img_idx.
        img_idxs = self.img_idxs
        select_idx = list(np.where(img_idxs.data.numpy() == img_idx))
        select_questions = self.input_questions[select_idx, :]
        return select_questions.squeeze(0)[:, 1:]  # removing <sos> token.

    def get_idx_to_token(self, questions=True):
        if questions:
            vocab = self.vocab_questions
        else:
            vocab = self.vocab_answers
        idx_to_token = dict(zip(list(vocab.values()), list(vocab.keys())))
        return idx_to_token

    def idx2word(self, seq_idx, delim=' ', stop_at_end=False, clean=False, ignored=["<SOS>"]):
        tokens = decode(seq_idx=seq_idx, idx_to_token=self.idx_to_token, stop_at_end=stop_at_end, delim=delim,
                        clean=clean, ignored=ignored)
        return tokens

    def word2idx(self, seq_tokens, allow_unk=True):
        idx = encode(seq_tokens=seq_tokens, token_to_idx=self.vocab_questions, allow_unk=allow_unk)
        return idx

    def __getitem__(self, index):
        input_question = self.input_questions[index, :]
        target_question = self.target_questions[index, :]
        img_idx = self.img_idxs[index]
        answer = self.answers[index]
        # loading img feature of img_idx
        feats = self.get_feats_from_img_idx(img_idx)

        return (input_question, target_question), feats, answer

    def __len__(self):
        if self.max_samples is None:
            return self.input_questions.size(0)
        else:
            return min(self.max_samples, self.input_questions.size(0))


if __name__ == '__main__':

    data_path = '../../data'
    vocab_path = os.path.join(data_path, "vocab.json")
    h5_questions_path = os.path.join(data_path, "train_questions.h5")
    h5_feats_path = os.path.join(data_path,
                                 "train_features.h5")  # Caution, here train_features.h5 corresponds only to the first 21 img of the train dataset.
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
    ep_questions = clevr_dataset.get_questions_from_img_idx(0)
    print('questions subset', ep_questions.shape)

    # -----------------------------------------------------------------------------
    # test max samples case.
    clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                  h5_feats_path=h5_feats_path,
                                  vocab_path=vocab_path,
                                  max_samples=21)
    num_samples = clevr_dataset.__len__()
    clevr_loader = DataLoader(dataset=clevr_dataset,
                              batch_size=1)

    for batch, ((inp, tar), feats, _) in enumerate(clevr_loader):
        if batch == 0:
            print('inp', inp[0, :])
            print('tar', tar[0, :])
            print('feats shape', feats.shape)
    print('number of samples', batch)
    # ----------------------------------------------- test get_questions_from_img_idx------------
    int = np.random.randint(0, 21, size=1)
    print(int)
    ep_questions = clevr_dataset.get_questions_from_img_idx(int).data.numpy()
    print('questions subset', ep_questions.shape)
    ep_questions = [list(ep_questions[i, :]) for i in range(ep_questions.shape[0])]
    decoded_questions = [clevr_dataset.idx2word(question, stop_at_end=True) for question in ep_questions]
    print('questions decoded :\n{}'.format("\n".join(decoded_questions)))
