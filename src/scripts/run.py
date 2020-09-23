import argparse
import datetime
import os

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.ppo import PPO
from agent.reinforce import REINFORCE
from envs.clevr_env import ClevrEnv
from models.rl_basic import PolicyLSTMBatch
from utils.utils_train import create_logger
from utils.utils_train import write_to_csv, compute_write_all_metrics


def get_writer(args, pre_trained, truncated, output_path):
    out_folder = "runs_{}_{}-{}-{}_{}_{}_len{}_debug{}_q{}_ent{}_k{}_b{}_lr{}_gradclip{}_trunc_{}_diffrew{}_fusion{}_losscorrection{}".format(
        args.agent,
        args.model,
        args.word_emb_size,
        args.hidden_size,
        pre_trained,
        truncated,
        args.max_len,
        args.debug,
        args.num_questions,
        args.entropy_coeff,
        args.num_truncated,
        args.update_every,
        args.lr,
        args.grad_clip,
        args.truncate_mode,
        args.diff_reward,
        args.fusion,
        args.is_loss_correction)

    if args.agent == 'PPO':
        out_folder = out_folder + '_eps{}_Kepochs{}'.format(args.eps_clip, args.K_epochs)

    if args.truncate_mode == 'proba_thr' and args.p_th is not None:
        out_folder = out_folder + '_pth{}'.format(args.p_th)

    if args.alpha_logits != 0:
        out_folder = out_folder + '_alpha-logits-{}'.format(args.alpha_logits)
    if args.alpha_decay_rate > 0:
        out_folder = out_folder + '_decay{}'.format(args.alpha_decay_rate)

    if args.truncate_mode is not None and args.epsilon_truncated > 0:
        out_folder = out_folder + '_eps-trunc{}'.format(args.epsilon_truncated)

    if args.train_policy == "truncated":
        out_folder = out_folder + '_truncated_policy'
    if args.reward == 'vqa':
        out_folder = out_folder + '_{}'.format(args.reward) + '_{}'.format(args.condition_answer) + '_mask-answers{}'.format(args.mask_answers)

    writer = SummaryWriter(log_dir=os.path.join(output_path, out_folder))
    return writer


def get_agent(pretrained_lm, writer, output_path, env, test_envs, policy):
    generic_kwargs = {"pretrained_lm": pretrained_lm,
                      "pretrain": args.pretrain,
                      "update_every": args.update_every,
                      "lr": args.lr,
                      "grad_clip": args.grad_clip, "writer": writer,
                      "truncate_mode": args.truncate_mode,
                      "num_truncated": args.num_truncated,
                      "p_th": args.p_th,
                      "out_path": output_path,
                      "log_interval": args.log_interval, "env": env,
                      "test_envs": test_envs,
                      "eval_no_trunc": args.eval_no_trunc,
                      "alpha_logits": args.alpha_logits,
                      "alpha_decay_rate": args.alpha_decay_rate,
                      "epsilon_truncated": args.epsilon_truncated,
                      "epsilon_truncated_rate": args.epsilon_truncated_rate,
                      "train_seed": args.train_seed,
                      "is_loss_correction": args.is_loss_correction}

    ppo_kwargs = {"policy": policy, "gamma": args.gamma,
                  "K_epochs": args.K_epochs,
                  "entropy_coeff": args.entropy_coeff,
                  "eps_clip": args.eps_clip}
    reinforce_kwargs = {"policy": policy, "gamma": args.gamma}
    algo_kwargs = {"PPO": ppo_kwargs, "REINFORCE": reinforce_kwargs}
    kwargs = {**algo_kwargs[args.agent], **generic_kwargs}

    agents = {"PPO": PPO, "REINFORCE": REINFORCE}

    agent = agents[args.agent](**kwargs)
    return agent


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, required=True,
                        help="data folder containing questions embeddings and img features")
    parser.add_argument("-out_path", type=str, required=True, help="out folder")
    # model args
    parser.add_argument('-model', type=str, default="lstm", help="model")
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-word_emb_size", type=int, default=8, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=24, help="dimension of the hidden state")
    parser.add_argument('-conv_kernel', type=int, default=1, help="conv kernel")
    parser.add_argument('-stride', type=int, default=2, help="stride conv")
    parser.add_argument('-num_filters', type=int, default=3, help="filters for conv")
    parser.add_argument('-fusion', type=str, default="cat", help="fusion mode")
    # RL algo args.
    parser.add_argument('-agent', type=str, default="PPO", help="RL agent")
    parser.add_argument('-K_epochs', type=int, default=10, help="# epochs of training each update_timestep")
    parser.add_argument('-update_every', type=int, default=20, help="update_every episode/timestep")
    parser.add_argument('-entropy_coeff', type=float, default=0.01, help="entropy coeff")
    parser.add_argument('-eps_clip', type=float, default=0.02, help="eps clip")
    parser.add_argument('-lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('-grad_clip', type=float, help="value of gradient norm clipping")
    parser.add_argument('-policy_path', type=str, default=None,
                        help="if specified, pre-trained model of the policy")
    # RL task args.
    parser.add_argument("-max_len", type=int, default=10, help="max episode length")
    parser.add_argument('-gamma', type=float, default=1., help="gamma")
    parser.add_argument('-reward', type=str, default="levenshtein_", help="type of reward function")
    parser.add_argument('-reward_path', type=str, help="path for the reward")
    parser.add_argument('-reward_vocab', type=str, help="vocab for the reward")

    parser.add_argument('-debug', type=str, default="0,69000",
                        help="debug mode: train on the first debug images")
    parser.add_argument('-num_questions', type=int, default=10, help="number of questions for each image")
    parser.add_argument('-diff_reward', type=int, default=0, help="is reward differential")
    parser.add_argument('-condition_answer', type=str, default="none",
                        help="type of answer condition, default to none")
    # truncation args.
    parser.add_argument('-lm_path', type=str, required=True,
                        help="the language model path (used for truncating the action space if truncate_mode is not None).Else, used only at test time")
    parser.add_argument('-truncate_mode', type=str,
                        help="truncation mode")  # arg that says now if are truncating the action space or not.
    parser.add_argument('-num_truncated', type=int, default=10, help="number of words from lm")
    parser.add_argument('-p_th', type=float,
                        help="probability threshold for proba threshold truncation mode")  # arg used in the proba_thr truncation function.
    parser.add_argument('-alpha_logits', default=0., type=float,
                        help="alpha value for the convex logits mixture. if 0, does not fuse the logits of the policy with the logits of the lm.")
    parser.add_argument('-alpha_decay_rate', default=0., type=float,
                        help="alpha decay rate for the convex logits mixture. if 0, does not decay the alpha")
    parser.add_argument('-epsilon_truncated', type=float, default=0.,
                        help="the agent sample from truncated or total action space")
    parser.add_argument('-epsilon_truncated_rate', type=float, default=1,
                        help="number of training iterations before epsilon truncated set to 1")
    parser.add_argument('-is_loss_correction', type=int, default=1,
                        help="adding the importance sampling ratio correction in the rl loss.")
    parser.add_argument('-train_policy', type=str, default="all_space",
                        help="train policy over all space or the truncated action space")  # arg to choose between trainig the complete policy or the truncated one in case of truncation.
    # train / test pipeline:
    parser.add_argument("-num_episodes_train", type=int, default=10, help="number of episodes training")
    parser.add_argument("-num_episodes_test", type=int, default=10, help="number of episodes test")
    parser.add_argument("-train_seed", type=int, default=0,
                        help="using a seed for the episode generation in training or not...")
    parser.add_argument('-resume_training', type=str, help='folder path to resume training from saved saved checkpoint')
    parser.add_argument('-eval_no_trunc', type=int, default=1,
                        help="if using truncation at training: at test time, evaluate also langage generated without truncation. Default to False.")
    parser.add_argument('-test_baselines', type=int, default=0, help="add test SL baselines for evaluation")
    # misc.
    parser.add_argument('-logger_level', type=str, default="INFO", help="level of logger")
    parser.add_argument('-log_interval', type=int, default=10, help="gamma")
    parser.add_argument('-pretrain', type=int, default=0, help="the agent use pretraining on the dataset")
    parser.add_argument('-mask_answers', type=int, default=0, help="mask answers")

    return parser


def run(args):
    type_folder = "train" if args.pretrain == 0 else "pretrain"
    if args.resume_training is not None:
        output_path = os.path.join(args.resume_training,
                                   "resume_training_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    else:
        output_path = os.path.join(args.out_path, "experiments", type_folder,
                                   "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    out_file_log = os.path.join(output_path, 'RL_training_log.log')
    out_policy_file = os.path.join(output_path, 'model.pth')

    logger = create_logger(out_file_log, level=args.logger_level)
    truncated = "basic" if args.truncate_mode is None else "truncated"
    pre_trained = "scratch" if args.policy_path is None else "pretrain"
    logger.info("RL from {} ...".format(pre_trained))
    if args.truncate_mode is not None:
        logger.info("with truncation...")
        if args.train_policy == 'truncated':
            logger.info("learning the truncated policy")
    else:
        logger.info("without truncation...")

    writer = get_writer(args, pre_trained, truncated, output_path)

    env = ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode="train", debug=args.debug,
                   num_questions=args.num_questions, diff_reward=args.diff_reward, reward_path=args.reward_path,
                   reward_vocab=args.reward_vocab, mask_answers=args.mask_answers)
    if args.reward == 'vqa':
        test_envs = [ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode=mode, debug=args.debug,
                              num_questions=args.num_questions, reward_path=args.reward_path,
                              reward_vocab=args.reward_vocab) for mode in
                     ["test_images"]]
    else:
        test_envs = [ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode=mode, debug=args.debug,
                              num_questions=args.num_questions, reward_path=args.reward_path,
                              reward_vocab=args.reward_vocab) for mode in
                     ["test_images", "test_text"]]

    pretrained_lm = None
    if args.lm_path is not None:
        pretrained_lm = torch.load(args.lm_path, map_location=torch.device('cpu'))
        pretrained_lm.eval()

    models = {"lstm": PolicyLSTMBatch}

    # creating the policy model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = models[args.model](env.clevr_dataset.len_vocab, args.word_emb_size, args.hidden_size,
                                kernel_size=args.conv_kernel,
                                stride=args.stride, num_filters=args.num_filters,
                                train_policy=args.train_policy, fusion=args.fusion, env=env,
                                condition_answer=args.condition_answer)
    if args.policy_path is not None:
        policy.load_state_dict(torch.load(args.policy_path, map_location=device), strict=False)
        # self.policy = torch.load(pretrained_policy, map_location=self.device)

    agent = get_agent(pretrained_lm, writer, output_path, env, test_envs, policy)

    eval_mode = ['sampling', 'greedy']  # TODO: put it as a parser arg.
    # eval_mode = ['greedy']

    if args.resume_training is not None:
        epoch, loss = agent.load_ckpt(os.path.join(args.resume_training, "checkpoints"))
        logger.info('resume training after {} episodes... current loss: {:2.2f}'.format(epoch, loss))
        agent.start_episode = epoch + 1
    if args.num_episodes_train > 0:  # trick to avoid a bug inside the agent.learn function in case of no training.
        agent.learn(num_episodes=args.num_episodes_train)
        agent.save(out_policy_file)
    else:
        logger.info("skipping training...")
    logger.info(
        '---------------------------------- STARTING EVALUATION --------------------------------------------------------------------------')
    for mode in eval_mode:
        logger.info(
            "----------------------------- Starting evaluation for {} action selection -------------------------".format(
                mode))
        agent.test(num_episodes=args.num_episodes_test, test_mode=mode, baselines=args.test_baselines)
    # write to csv test scalar metrics:
    if agent.truncate_mode is not None and args.eval_no_trunc:
        logger.info("computing all metrics for dialog keeping the truncation mask...")
        compute_write_all_metrics(agent=agent, output_path=output_path, logger=logger, keep="with_trunc")
        logger.info("computing all metrics for dialog without the truncation mask...")
        compute_write_all_metrics(agent=agent, output_path=output_path, logger=logger, keep="no_trunc")
    else:
        compute_write_all_metrics(agent=agent, output_path=output_path,
                                  logger=logger, keep=None)
    logger.info(
        '------------------------------------ DONE ---------------------------------------------------------------')
    return agent


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
