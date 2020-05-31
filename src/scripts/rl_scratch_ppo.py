import argparse
import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from agent.ppo import PPO
from envs.clevr_env import VectorEnv, ClevrEnv
from models.rl_basic import PolicyGRU_Custom, PolicyGRUWordBatch, PolicyLSTMWordBatch
from utils.utils_train import create_logger

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-word_emb_size", type=int, default=12, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=24, help="dimension of the hidden state")
    parser.add_argument("-max_len", type=int, default=10, help="max episode length")
    # parser.add_argument("-num_training_steps", type=int, default=1000, help="number of training_steps")
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
    parser.add_argument('-model', type=str, default="gru_word", help="model")
    parser.add_argument('-K_epochs', type=int, default=5, help="# epochs of training each update_timestep")
    parser.add_argument('-update_timestep', type=int, default=100, help="update_timestep")
    parser.add_argument('-entropy_coeff', type=float, default=0.01, help="entropy coeff")
    parser.add_argument('-eps_clip', type=float, default=0.2, help="eps clip")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = os.path.join(args.out_path, "experiments", "train",
                               "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    out_file_log = os.path.join(output_path, 'RL_training_log.log')

    logger = create_logger(out_file_log, level=args.logger_level)

    writer = SummaryWriter(log_dir=os.path.join(output_path, 'runs'))

    env = ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode="train", debug=True)

    #make_env_fn = lambda: ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode="train", debug=True)
    #envs = VectorEnv(make_env_fn, n=2)

    models = {"gru_word": PolicyGRUWordBatch,
              "gru": PolicyGRU_Custom,
              "lstm": PolicyLSTMWordBatch}

    policy = models[args.model](env.clevr_dataset.len_vocab, args.word_emb_size, args.hidden_size)
    policy_old = models[args.model](env.clevr_dataset.len_vocab, args.word_emb_size, args.hidden_size)
    policy_old.load_state_dict(policy.state_dict())

    agent = PPO(policy=policy, policy_old=policy_old, env=env, gamma=args.gamma, K_epochs=args.K_epochs,
                update_timestep=args.update_timestep, entropy_coeff=args.entropy_coeff, eps_clip=args.eps_clip)

    agent.learn(log_interval=args.log_interval, num_episodes=args.num_episodes_train,
                writer=writer, output_path=output_path)
