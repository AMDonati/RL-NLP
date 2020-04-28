import torch
import h5py
import argparse
import time
import os
import json
import numpy as np
from models.LM_networks import GRUModel

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
  parser.add_argument("-emb_size", type=int, default=512, help="dimension of the embedding layer")
  parser.add_argument("-hidden_size", type=int, default=128, help="dimension of the hidden state")
  parser.add_argument("-p_drop", type=float, default=0, help="dropout rate")
  parser.add_argument("-data_path", type=str, default='../../data/CLEVR_v1.0/temp')
  parser.add_argument('-cuda', type=bool, default=False, help='use cuda')

  args = parser.parse_args()

  device = torch.device("cpu" if not args.cuda else "cuda")

  ###############################################################################
  # Load data
  ###############################################################################

  train_questions_path = os.path.join(args.data_path, "train_questions_subset.h5")
  val_questions_path = os.path.join(args.data_path, "val_questions_subset.h5")
  test_questions_path = os.path.join(args.data_path, "test_questions_subset.h5")

  def get_questions(h5_path):
    hf = h5py.File(h5_path, 'r')
    questions = hf.get('questions')
    questions = np.array(questions)
    questions = torch.tensor(questions, dtype=torch.int)
    return questions

  train_questions = get_questions(train_questions_path)
  val_questions = get_questions(val_questions_path)
  test_questions = get_questions(test_questions_path)

  # load vocab
  vocab_path = os.path.join(args.data_path, "vocab_subset_from_train.json")
  with open(vocab_path, 'r') as f:
    vocab = json.load(f)['question_token_to_idx']
  num_tokens = len(vocab)

  # transform data into batches of data #TODO: later on, use torch.Data module.
  def batchify(data, batch_size):
    num_batches = data.size(0) // batch_size
    # Trim off remainders.
    data = data.narrow(0,0,num_batches * batch_size)
    # divide the data across the batches
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)

  BATCH_SIZE = 64
  train_data = batchify(train_questions, BATCH_SIZE)

  ###############################################################################
  # Build the model
  ###############################################################################

  model = GRUModel(num_tokens=num_tokens,
                   emb_size=args.emb_size,
                   hidden_size=args.hidden_size,
                   num_layers=args.num_layers,
                   p_drop=args.p_drop) #TODO: add to(device)?

  ###############################################################################
  # Train the model
  ###############################################################################
  def train(num_tokens):
    model.train() # Turns on train mode which enables dropout.
    total_loss = 0.
    start_time = time.time()

    # loop over batches

