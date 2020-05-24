import argparse
import datetime
import logging
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from agent.reinforce import REINFORCE
from envs.clevr_env import ClevrEnv
from eval.rl import test
from models.rl_basic import PolicyGRU
from train.rl import train
from utils.utils_train import create_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-word_emb_size", type=int, default=12, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=24, help="dimension of the hidden state")
    parser.add_argument("-max_len", type=int, default=10, help="max episode length")
    parser.add_argument("-num_episodes_train", type=int, default=2000, help="number of episodes training")
    parser.add_argument("-num_episodes_test", type=int, default=100, help="number of episodes test")

    parser.add_argument("-data_path", type=str, required=True,
                        help="data folder containing questions embeddings and img features")
    parser.add_argument("-out_path", type=str, required=True, help="out folder")
    parser.add_argument('-logger_level', type=str, default="INFO", help="level of logger")
    parser.add_argument('-gamma', type=float, default=1., help="gamma")
    parser.add_argument('-log_interval', type=int, default=10, help="gamma")
    parser.add_argument('-reward', type=str, default="cosine", help="type of reward function")
    parser.add_argument('-lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('-model', type=str, default="gru", help="model")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = os.path.join(args.out_path, "lm", "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    out_file_log = os.path.join(output_path, 'RL_training_log.log')

    logger = create_logger(out_file_log, level=args.logger_level)

    writer = SummaryWriter(log_dir=os.path.join(output_path, 'runs'))

    env = ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode="train")

    models = {"gru": PolicyGRU}

    model = models[args.model](env.clevr_dataset.len_vocab, args.word_emb_size, args.hidden_size)

    # Training language model
    pretraining_model = model.to(device)
    pretraining_agent = REINFORCE(model=pretraining_model, gamma=args.gamma, lr=args.lr)
    _, saved_path = train(env=env, agent=pretraining_agent, log_interval=args.log_interval,
                          num_episodes=args.num_episodes_train,
                          pretrain=True, writer=writer, output_path=output_path)

    logging.info("-" * 20)
    logging.info("TEST")
    logging.info("-" * 20)

    # using val set because no answer in test set -> bug
    env = ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode="val")
    test(env=env, agent=pretraining_agent, num_episodes=args.num_episodes_test,
         writer=writer, saved_path=saved_path)
