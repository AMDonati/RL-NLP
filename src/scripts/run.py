import argparse
import datetime
import os

import torch
from torch.utils.tensorboard import SummaryWriter

from agent.ppo import PPO
from agent.reinforce import REINFORCE
from envs.clevr_env import ClevrEnv
from models.rl_basic import PolicyLSTMWordBatch, PolicyLSTMBatch
from utils.utils_train import create_logger

if __name__ == '__main__':
    # -data_path /Users/guillaumequispe/PycharmProjects/RL-NLP/data -out_path /Users/guillaumequispe/PycharmProjects/RL-NLP/output
    # -max_len 7 -logger_level DEBUG -num_episodes_train 4000 -log_interval 1 -reward "levenshtein_"
    # -model lstm_word -update_timestep 50 -K_epochs 10 -entropy_coeff 0.01 -eps_clip 0.02
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
    parser.add_argument('-pretrained_path', type=str, default=None,
                        help="if specified, the language model truncate the action space")
    parser.add_argument('-pretrain', type=int, default=0, help="the agent use pretraining on the dataset")
    parser.add_argument('-debug', type=int, default=1,
                        help="debug mode: train on just one question from the first image")
    parser.add_argument('-agent', type=str, default="PPO", help="RL agent")
    parser.add_argument('-conv_kernel', type=int, default=1, help="conv kernel")
    parser.add_argument('-stride', type=int, default=2, help="stride conv")
    parser.add_argument('-num_filters', type=int, default=3, help="filters for conv")
    parser.add_argument('-num_truncated', type=int, default=10, help="number of words from lm")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = os.path.join(args.out_path, "experiments", "train",
                               "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    out_file_log = os.path.join(output_path, 'RL_training_log.log')
    out_policy_file = os.path.join(output_path, 'model.pth')

    logger = create_logger(out_file_log, level=args.logger_level)

    writer = SummaryWriter(log_dir=os.path.join(output_path, "runs_{}_{}_{}_{}_{}".format(args.model, args.eps_clip,
                                                                                          args.num_filters, args.stride,
                                                                                          args.conv_kernel)))

    env = ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode="train", debug=args.debug)

    # make_env_fn = lambda: ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode="train", debug=True)
    # envs = VectorEnv(make_env_fn, n=2)

    pretrained_lm = None
    if args.pretrained_path is not None:
        pretrained_lm = torch.load(args.pretrained_path)
        pretrained_lm.eval()

    models = {
        "lstm": PolicyLSTMBatch,
        "lstm_word": PolicyLSTMWordBatch}

    generic_kwargs = {"pretrained_lm": pretrained_lm, "pretrain": args.pretrain, "word_emb_size": args.word_emb_size,
                      "hidden_size": args.hidden_size, "kernel_size": args.conv_kernel, "stride": args.stride,
                      "num_filters": args.num_filters, "num_truncated": args.num_truncated}

    ppo_kwargs = {"policy": models[args.model], "env": env, "gamma": args.gamma,
                  "K_epochs": args.K_epochs,
                  "update_timestep": args.update_timestep, "entropy_coeff": args.entropy_coeff,
                  "eps_clip": args.eps_clip}
    reinforce_kwargs = {"env": env, "policy": models[args.model], "gamma": args.gamma, "lr": args.lr,
                        "word_emb_size": args.word_emb_size, "hidden_size": args.hidden_size}
    algo_kwargs = {"PPO": ppo_kwargs, "REINFORCE": reinforce_kwargs}
    kwargs = {**algo_kwargs[args.agent], **generic_kwargs}

    agents = {"PPO": PPO, "REINFORCE": REINFORCE}

    agent = agents[args.agent](**kwargs)

    agent.learn(log_interval=args.log_interval, num_episodes=args.num_episodes_train,
                writer=writer, output_path=output_path)
    agent.save(out_policy_file)
    agent.test(log_interval=args.log_interval, num_episodes=args.num_episodes_test, writer=writer)
