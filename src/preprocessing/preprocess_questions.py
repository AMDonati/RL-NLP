#!/usr/bin/env python3

# Copyright 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import json
import h5py
import numpy as np
import os

from preprocessing.text_functions import tokenize, encode, build_vocab

"""
Preprocessing script for CLEVR question files.
"""
def extract_short_json(json_data_path, out_path, num_questions):

  with open(json_data_path, 'r') as f:
    questions = json.load(f)['questions']
  select_questions = questions[:num_questions]
  out_json = {'questions': select_questions}

  with open(out_path, 'w') as f:
    json.dump(out_json, f)

def preprocess_questions(min_token_count, punct_to_keep, punct_to_remove, add_start_token, add_end_token, json_data_path, vocab_out_path, h5_out_path):

  print('Loading Data...')
  with open(json_data_path, 'r') as f:
    questions = json.load(f)['questions']

  print('number of questions in json file: {}'.format(len(questions)))

  if os.path.isfile(vocab_out_path):
    print('Loading vocab...')
    with open(vocab_out_path, 'r') as f:
      vocab = json.load(f)
  else:
    print('Building vocab...')
    list_questions = [q['question'] for q in questions]
    question_token_to_idx = build_vocab(sequences=list_questions,
                                          min_token_count=min_token_count,
                                          punct_to_keep=punct_to_keep,
                                          punct_to_remove=punct_to_remove)
    print('number of words in vocab: {}'.format(len(question_token_to_idx)))
    vocab = {'question_token_to_idx': question_token_to_idx}

    with open(vocab_out_path, 'w') as f:
        json.dump(vocab, f)

  print('Encoding questions...')
  questions_encoded = []
  for orig_idx, q in enumerate(questions):
    question = q['question']

    question_tokens = tokenize(s=question,
                              punct_to_keep=punct_to_keep,
                              punct_to_remove=punct_to_remove,
                              add_start_token=add_start_token,
                              add_end_token=add_end_token)
    question_encoded = encode(seq_tokens=question_tokens,
                              token_to_idx=vocab['question_token_to_idx'],
                              allow_unk=True)
    questions_encoded.append(question_encoded)

  # Pad encoded questions
  max_question_length = max(len(x) for x in questions_encoded)
  for qe in questions_encoded:
    while len(qe) < max_question_length:
      qe.append(vocab['question_token_to_idx']['<PAD>'])

  # Create h5 file
  print('Writing output...')
  questions_encoded = np.asarray(questions_encoded, dtype=np.int32)
  print(questions_encoded.shape)
  with h5py.File(h5_out_path, 'w') as f:
    f.create_dataset('questions', data=questions_encoded)


if __name__ == '__main__':

  train_json_data_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/questions/CLEVR_train_questions.json'
  train_out_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/temp/train_questions_subset.json'
  extract_short_json(json_data_path=train_json_data_path, out_path=train_out_path, num_questions=5000)

  val_json_data_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/questions/CLEVR_val_questions.json'
  val_out_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/temp/val_questions_subset.json'
  extract_short_json(json_data_path=val_json_data_path, out_path=val_out_path, num_questions=2000)

  test_json_data_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/questions/CLEVR_test_questions.json'
  test_out_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/temp/test_questions_subset.json'
  extract_short_json(json_data_path=test_json_data_path, out_path=test_out_path, num_questions=2000)

  vocab_out_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/temp/vocab_subset_from_train.json'
  punct_to_keep = [';', ',', '?']
  punct_to_remove = ['.']

  # ------------------ preprocessing for train dataset ---------------------------------------------------------------------------------

  json_data_path = train_out_path
  h5_out_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/temp/train_questions_subset.h5'

  preprocess_questions(min_token_count=1,
                       punct_to_keep=punct_to_keep,
                       punct_to_remove=punct_to_remove,
                       add_start_token=True,
                       add_end_token=True,
                       json_data_path=json_data_path,
                       vocab_out_path=vocab_out_path,
                       h5_out_path=h5_out_path)

  # ------------------- preprocessing for val dataset ----------------------------------------------------------------------------------

  json_data_path = val_out_path
  h5_out_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/temp/val_questions_subset.h5'

  preprocess_questions(min_token_count=1,
                       punct_to_keep=punct_to_keep,
                       punct_to_remove=punct_to_remove,
                       add_start_token=True,
                       add_end_token=True,
                       json_data_path=json_data_path,
                       vocab_out_path=vocab_out_path,
                       h5_out_path=h5_out_path)

  # ----------------- preprocessing for test dataset ------------------------------------------------------------------------------------
  json_data_path = test_out_path
  h5_out_path = '/Users/alicemartin/000_Boulot_Polytechnique/07_PhD_thesis/code/RL-NLP/data/CLEVR_v1.0/temp/test_questions_subset.h5'

  preprocess_questions(min_token_count=1,
                       punct_to_keep=punct_to_keep,
                       punct_to_remove=punct_to_remove,
                       add_start_token=True,
                       add_end_token=True,
                       json_data_path=json_data_path,
                       vocab_out_path=vocab_out_path,
                       h5_out_path=h5_out_path)