# https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3
import torch
import argparse
import time
import os
import math
import json
from models.LM_networks import GRUModel
from preprocessing.QuestionsDataset import QuestionsDataset
from torch.utils.data import DataLoader
from utils.utils_train import create_logger, write_to_csv

'''
training script for LM network. 
Inspired from: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
'''

if __name__ == '__main__':

  parser = argparse.ArgumentParser()

  parser.add_argument("-num_layers", type=int, required=True, default=1, help="num layers for language model")
  parser.add_argument("-emb_size", type=int, required=True, default=12, help="dimension of the embedding layer")
  parser.add_argument("-hidden_size", type=int, required=True, default=24, help="dimension of the hidden state")
  parser.add_argument("-p_drop", type=float, required=True, default=0, help="dropout rate")
  parser.add_argument("-grad_clip", type=bool, required=True, default=False)
  parser.add_argument("-data_path", type=str, required=True, default='../../data')
  parser.add_argument("-out_path", type=str, required=True, default='../../output')
  parser.add_argument('-cuda', type=bool, required=True, default=False, help='use cuda')

  args = parser.parse_args()

  device = torch.device("cpu" if not args.cuda else "cuda")
  #device = torch.device("cpu")

  ###############################################################################
  # Load data
  ###############################################################################

  train_questions_path = os.path.join(args.data_path, "train_questions.h5")
  val_questions_path = os.path.join(args.data_path, "val_questions.h5")
  test_questions_path = os.path.join(args.data_path, "test_questions.h5")
  vocab_path = os.path.join(args.data_path, "vocab.json")

  train_dataset = QuestionsDataset(h5_questions_path=train_questions_path, vocab_path=vocab_path)
  val_dataset = QuestionsDataset(h5_questions_path=val_questions_path, vocab_path=vocab_path)
  test_dataset = QuestionsDataset(h5_questions_path=test_questions_path, vocab_path=vocab_path)

  num_tokens = train_dataset.vocab_len
  BATCH_SIZE = 512
  PAD_IDX = train_dataset.get_vocab()["<PAD>"]

  train_generator = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, drop_last=True)
  val_generator = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, drop_last=True)
  test_generator = DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE, drop_last=True)

  ###############################################################################
  # Build the model
  ###############################################################################

  model = GRUModel(num_tokens=num_tokens,
                   emb_size=args.emb_size,
                   hidden_size=args.hidden_size,
                   num_layers=args.num_layers,
                   p_drop=args.p_drop).to(device)
  learning_rate = 0.001
  optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
  criterion = torch.nn.NLLLoss(ignore_index=PAD_IDX)
  EPOCHS = 50

  ###############################################################################
  # Create logger, output_path and config file.
  ###############################################################################

  out_path = 'GRU_layers_{}_emb_{}_hidden_{}_pdrop_{}_gradclip_{}'.format(args.num_layers, args.emb_size, args.hidden_size, args.p_drop, args.grad_clip)
  out_path = os.path.join(args.out_path, out_path)
  if not os.path.exists(out_path):
    os.makedirs(out_path)
  out_file_log = os.path.join(out_path, 'training_log.log')
  logger = create_logger(out_file_log)
  out_csv = os.path.join(out_path, 'train_history.csv')
  model_path = os.path.join(out_path, 'model.pt')
  config_path = os.path.join(out_path, 'config.json')

  hparams = {}
  hparams["model"] = "GRU"
  hparams["emb_size"] = args.emb_size
  hparams["hidden_size"] = args.hidden_size
  hparams["p_drop"] = args.p_drop
  hparams["grad_clip"] = args.grad_clip
  hparams["BATCH_SIZE"] = BATCH_SIZE
  hparams["learning_rate"] = learning_rate
  config = {"hparams": hparams}

  with open(config_path, mode='w') as f:
    json.dump(config, f)

  ###############################################################################
  # Training functions
  ###############################################################################

  def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history.
    so that each hidden_state h_t is detached from the backprop graph once used. """
    if isinstance(h, torch.Tensor):
      return h.detach()
    else:
      return tuple(repackage_hidden(v) for v in h)

  def train_one_epoch(model, train_generator, optimizer, criterion, BATCH_SIZE, grad_clip):

    model.train() # Turns on train mode which enables dropout.
    hidden = model.init_hidden(BATCH_SIZE)
    total_loss = 0.
    start_time = time.time()

    # loop over batches
    for batch, (inputs, targets) in enumerate(train_generator):
      inputs, targets = inputs.to(device), targets.to(device)
      inputs, targets = inputs.long().t(), targets.view(targets.size(1)*targets.size(0)).long() # inputs: (S,B) # targets: (S*B)
      optimizer.zero_grad()
      hidden = repackage_hidden(hidden)
      output, hidden = model(inputs, hidden) # output (S * B, V), hidden (S,B,1)
      loss = criterion(output, targets)
      loss.backward()

      # clip grad norm:
      if grad_clip:
        torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=0.25)
      optimizer.step()
      total_loss += loss.item()

    curr_loss = total_loss / (batch + 1)
    elapsed = time.time() - start_time

    return curr_loss, elapsed

  def evaluate(model, val_generator, criterion, BATCH_SIZE):
    model.eval() # turn on evaluation mode which disables dropout.
    total_loss = 0.
    hidden = model.init_hidden(BATCH_SIZE)
    with torch.no_grad():
      for batch, (inputs, targets) in enumerate(val_generator):
        inputs, targets = inputs.to(device), targets.to(device)
        inputs, targets = inputs.long().t(), targets.view(targets.size(1)*targets.size(0)).long()
        output, hidden = model(inputs, hidden)
        hidden = repackage_hidden(hidden)
        total_loss += criterion(output, targets).item()

    return total_loss / (batch + 1)

  ################################################################################################################################################
  # Train the model
  #################################################################################################################################################

  logger.info("start training...")
  logger.info("hparams: {}".format(hparams))
  train_loss_history, train_ppl_history, val_loss_history, val_ppl_history = [], [], [], []
  best_val_loss = None
  for epoch in range(EPOCHS):
    logger.info('epoch {}/{}'.format(epoch+1, EPOCHS))
    train_loss, elapsed = train_one_epoch(model=model, train_generator=train_generator, optimizer=optimizer, criterion=criterion, BATCH_SIZE=BATCH_SIZE, grad_clip=args.grad_clip)
    logger.info('train loss {:5.3f} - train perplexity {:8.3f}'.format(train_loss, math.exp(train_loss)))
    train_loss_history.append(train_loss)
    train_ppl_history.append(math.exp(train_loss))
    logger.info('time for one epoch...{:5.2f}'.format(elapsed))
    val_loss = evaluate(model=model, val_generator=val_generator, criterion=criterion, BATCH_SIZE=BATCH_SIZE)
    logger.info('val loss: {:5.3f} - val perplexity: {:8.3f}'.format(val_loss, math.exp(val_loss)))
    val_loss_history.append(val_loss)
    val_ppl_history.append(math.exp(val_loss))
    logger.info('-' * 89)
    # Save the model if the validation loss is the best we've seen so far.
    if not best_val_loss or val_loss < best_val_loss:
      with open(model_path, 'wb') as f:
        torch.save(model, f)
      best_val_loss = val_loss

  logger.info("saving loss and metrics information...")
  hist_keys = ['train_loss', 'train_ppl', 'val_loss', 'val_ppl']
  hist_dict = dict(zip(hist_keys, [train_loss_history, train_ppl_history, val_loss_history, val_ppl_history]))
  write_to_csv(out_csv, hist_dict)




