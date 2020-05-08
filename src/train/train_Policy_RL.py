# example of ROUGE computation: https://github.com/atulkum/pointer_summarizer/blob/master/data_util/utils.py
import torch
from models.Policy_network import PolicyLSTM
from data_provider.CLEVR_Dataset import CLEVR_Dataset
import numpy as np
import argparse, os, math, statistics
import random
from collections import namedtuple
from torch.distributions import Categorical
from utils.utils_train import create_logger

#  trick for boolean parser args.
def str2bool(v):
  if isinstance(v, bool):
    return v
  if v.lower() in ('yes', 'true', 't', 'y', '1'):
    return True
  elif v.lower() in ('no', 'false', 'f', 'n', '0'):
    return False
  else:
    raise argparse.ArgumentTypeError('Boolean value expected.')


def select_action(policy_network, state, device):
  policy_network.train()
  hidden = policy_network.init_hidden(1)
  #with torch.no_grad():
  state.text.to(device)
  state.img.to(device)
  probas, _ = policy_network(state.text, state.img, hidden) # probas > shape (s, num_tokens)
  m = Categorical(probas[-1,:])  # multinomial distribution with weights = probas.
  action = m.sample()
  log_prob = m.log_prob(action)
  return action.view(1,1), log_prob # action and log_prob of shape (1).

def get_reward(next_state_text, ep_questions, EOS_idx):
  # remove <EOS> token if needed.
  # if next_state_text[-1] == EOS_idx:
  #   next_state_text = next_state_text[:-1]
  # dialog = next_state_text.data.numpy()
  # ep_questions = ep_questions.data.numpy()
  # bools = []
  # for i in range(ep_questions.size(1)):
  #   question = ep_questions[:, i]
  #   if len(question) == len(dialog):
  #     bool = np.array_equal(question, dialog)
  #   else:
  #     bool = False
  #   bools.append(bool)  # TODO np.any()?
  return 0.


# function generate one episode. debugged.
#TODO: batchify this function.
def generate_one_episode(clevr_dataset, policy_network, special_tokens, device, seed=None):
  max_length = clevr_dataset.input_questions.size(0)  # max_length set-up to max length of questions dataset.
  max_length = 5 # for debugging.
  # sample initial state
  if seed is not None:
    np.random.seed = seed
  img_idx = np.random.randint(0, len(clevr_dataset.img_idxs))
  img_idx = 10 # for debugging.
  ep_GD_questions = clevr_dataset.get_questions_from_img_idx(img_idx) # shape (max_len - 1, 10) # used to compute the final reward of the episode.
  img_feats = clevr_dataset.get_feats_from_img_idx(img_idx) # shape (1024, 14, 14)
  initial_state = State(torch.LongTensor([special_tokens.SOS_idx]).view(1,1), img_feats.unsqueeze(0))

  state = initial_state
  done = False
  step = 0
  rewards, log_probs = [], []
  while not done:
    # select the next action from the state using an epsilon greedy policy:
    action, log_prob = select_action(policy_network, state, device)
    # compute next state, done, reward from the action.
    next_state = State(torch.cat([state.text, action]), state.img)
    done = True if action.item() == special_tokens.EOS_idx or step == (max_length - 1) else False
    if done:
      reward = get_reward(next_state_text=next_state.text, ep_questions=ep_GD_questions, EOS_idx=special_tokens.EOS_idx)
    else:
      reward = 0
      step += 1
    rewards.append(reward)
    log_probs.append(log_prob)
    state = next_state

  episode = Episode(img_idx, img_feats, ep_GD_questions, state.text, rewards)

  return_ep = sum(rewards)
  returns = [return_ep] * (step + 1)

  return log_probs, returns, episode


def padder_batch(batch):
  len_episodes = [len(l) for l in batch]
  max_len = max(len_episodes)  # finds the maximal length of episodes.
  batch_tensors = [torch.FloatTensor(l).unsqueeze(-1) for l in batch]  # tensors of shape (len_ep, 1)
  batch_tensors_padded = [torch.cat([t, torch.zeros(max_len - len, t.size(-1), dtype=t.dtype)]) for (t, len) in
                          zip(batch_tensors, len_episodes)] # TODO: use masked_fill_(mask, value) instead ?
  batch = torch.stack(batch_tensors_padded, dim=0)
  return batch


def train_episodes_batch(log_probs_batch, returns_batch, optimizer):
  # normalize returns
  eps = np.finfo(np.float32).eps.item()
  returns_batch = (returns_batch - returns_batch.mean()) / (returns_batch.std() + eps)
  reinforce_loss = -log_probs_batch * returns_batch  # shape (batch_size, max_episode_len, 1) # opposite of REINFORCE objective function to apply a gradient descent algo.
  reinforce_loss = reinforce_loss.squeeze(-1).sum(dim=1).mean(dim=0)
  optimizer.zero_grad()
  reinforce_loss.backward() # ERROR here: no grad_fn on the loss... # grad_fn is None on log_probs_batch and returns_batch.
  optimizer.step()

  return reinforce_loss.item()


def REINFORCE(train_dataset, policy_network, special_tokens, batch_size, num_training_steps, optimizer, device, logger, log_interval=100, store_episodes=True):
  running_return = 0.
  all_episodes = []
  for i in range(num_training_steps):
    log_probs_batch, returns_batch, episodes_batch = [], [], []
    for _ in range(batch_size):
      log_probs, returns, episode = generate_one_episode(clevr_dataset=train_dataset,
                                                         policy_network=policy_network,
                                                         special_tokens=special_tokens,
                                                         device=device)
      log_probs_batch.append(log_probs)
      returns_batch.append(returns)
      if store_episodes:
        episodes_batch.append(episode)

    batch_avg_return = statistics.mean([r[-1] for r in returns_batch])
    log_probs_batch, returns_batch = padder_batch(log_probs_batch), padder_batch(returns_batch)
    loss = train_episodes_batch(log_probs_batch=log_probs_batch, returns_batch=returns_batch, optimizer=optimizer)
    running_return = 0.1 * batch_avg_return + (1 - 0.1) * running_return
    if i % log_interval == 0:
      logger.info('train loss for training step {}: {:5.3f}'.format(i, loss))
      logger.info('running return for training step {}: {:8.3f}'.format(i, loss))
    if store_episodes:
      all_episodes.append(episodes_batch)

  return all_episodes

if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
  parser.add_argument("-word_emb_size", type=int, default=12, help="dimension of the embedding layer")
  parser.add_argument("-hidden_size", type=int, default=24, help="dimension of the hidden state")
  parser.add_argument("-p_drop", type=float, default=0, help="dropout rate")
  parser.add_argument("-grad_clip", type=float)
  parser.add_argument("-lr", type=float, default=0.001)
  parser.add_argument("-bs", type=int, default=16, help="batch size")
  parser.add_argument("-num_training_steps", type=int, default=1000, help="number of training_steps")
  parser.add_argument("-data_path", type=str, required=True, help="data folder containing questions embeddings and img features")
  parser.add_argument("-out_path", type=str, required=True, help="out folder")
  parser.add_argument('-num_workers', type=int, default=0, help="num workers for DataLoader")

  args = parser.parse_args()
  device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

  ###############################################################################
  # Build CLEVR DATASET
  ###############################################################################

  h5_questions_path = os.path.join(args.data_path, 'train_questions.h5')
  h5_feats_path = os.path.join(args.data_path, 'train_features.h5')
  vocab_path = os.path.join(args.data_path, 'vocab.json')
  clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                h5_feats_path=h5_feats_path,
                                vocab_path=vocab_path)

  num_tokens = clevr_dataset.len_vocab
  feats_shape = clevr_dataset.feats_shape
  SOS_idx = clevr_dataset.vocab_questions["<SOS>"]
  EOS_idx = clevr_dataset.vocab_questions["<EOS>"]

  Special_Tokens = namedtuple('Special_Tokens', ('SOS_idx', 'EOS_idx'))
  special_tokens = Special_Tokens(SOS_idx, EOS_idx)
  State = namedtuple('State', ('text', 'img'))
  Episode = namedtuple('Episode', ('img_idx', 'img_feats', 'GD_questions', 'dialog', 'rewards')) #TODO: Build an Episode Dataset instead.

  ###############################################################################
  # Build the Policy Network and define hparams
  ###############################################################################

  policy_network = PolicyLSTM(num_tokens=num_tokens,
                              word_emb_size=args.word_emb_size,
                              emb_size=args.word_emb_size + feats_shape[0]*feats_shape[1]*feats_shape[2],
                              hidden_size=args.hidden_size,
                              num_layers=args.num_layers,
                              p_drop=args.p_drop)

  optimizer = torch.optim.Adam(policy_network.parameters(), lr=args.lr)

  out_file_log = os.path.join(args.out_path, 'RL_training_log.log')
  logger = create_logger(out_file_log)

  # -------- test of generate one episode function  ------------------------------------------------------------------------------------------------------
  log_probs, returns, episodes = generate_one_episode(clevr_dataset=clevr_dataset,
                                                      policy_network=policy_network,
                                                      special_tokens=special_tokens,
                                                      device=device)
  #--------- test of REINFORCE function --------------------------------------------------------------------------------------------------------------------
  all_episodes = REINFORCE(train_dataset=clevr_dataset,
                           policy_network=policy_network,
                           special_tokens=special_tokens,
                           batch_size=args.bs,
                           optimizer=optimizer,
                           device=device,
                           num_training_steps=args.num_training_steps,
                           logger=logger)