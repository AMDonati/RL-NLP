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


def run(args):
    type_folder = "train" if args.pretrain == 0 else "pretrain"
    if args.resume_training is not None:
        output_path = args.resume_training
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
        logger.info("with truncation")
    else:
        logger.info("without truncation")
    out_folder = "runs_{}_{}-{}-{}_{}_{}_len{}_debug{}_q{}_ent{}_k{}_b{}_lr{}_eps-adam{}_gradclip{}_trunc_{}_diffrew{}".format(
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
        args.eps,
        args.grad_clip,
        args.truncate_mode,
        args.diff_reward)

    if args.agent == 'PPO':
        out_folder = out_folder + '_eps{}_Kepochs{}'.format(args.eps_clip, args.K_epochs)

    writer = SummaryWriter(log_dir=os.path.join(output_path,
                                                out_folder))

    envs = [ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode=mode, debug=args.debug,
                     num_questions=args.num_questions, diff_reward=args.diff_reward) for mode in
            ["train", "test_images", "test_text"]]

    pretrained_lm = None
    if args.lm_path is not None:
        pretrained_lm = torch.load(args.lm_path, map_location=torch.device('cpu'))
        pretrained_lm.eval()

    models = {
        "lstm": PolicyLSTMBatch,
        "lstm_word": PolicyLSTMWordBatch}

    # creating the policy model.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = models[args.model](envs[0].clevr_dataset.len_vocab, args.word_emb_size, args.hidden_size,
                                kernel_size=args.conv_kernel,
                                stride=args.stride, num_filters=args.num_filters, rl=True)
    if args.policy_path is not None:
        policy.load_state_dict(torch.load(args.policy_path, map_location=device), strict=False)
        # self.policy = torch.load(pretrained_policy, map_location=self.device)

    generic_kwargs = {"pretrained_lm": pretrained_lm, "lm_sl": args.lm_sl,
                      "pretrain": args.pretrain,
                      "update_every": args.update_every,
                      "lr": args.lr,
                      "eps": args.eps,
                      "grad_clip": args.grad_clip, "writer": writer,
                      "truncate_mode": args.truncate_mode,
                      "num_truncated": args.num_truncated,
                      "k_min": args.k_min,
                      "p_th": args.p_th,
                      "out_path": output_path,
                      "log_interval": args.log_interval, "env": envs[0],
                      "test_envs": envs}

    ppo_kwargs = {"policy": policy, "gamma": args.gamma,
                  "K_epochs": args.K_epochs,
                  "entropy_coeff": args.entropy_coeff,
                  "eps_clip": args.eps_clip}
    reinforce_kwargs = {"policy": policy, "gamma": args.gamma}
    algo_kwargs = {"PPO": ppo_kwargs, "REINFORCE": reinforce_kwargs}
    kwargs = {**algo_kwargs[args.agent], **generic_kwargs}

    agents = {"PPO": PPO, "REINFORCE": REINFORCE}

    agent = agents[args.agent](**kwargs)

    eval_mode = ['sampling', 'greedy'] #TODO: put it as a parser arg.
    #eval_mode = ['greedy']

    if args.resume_training is not None:
        epoch, loss = agent.load_ckpt()
        logger.info('resume training after {} episodes... current loss: {:2.2f}'.format(epoch, loss))
        agent.start_episode = epoch
    agent.learn(num_episodes=args.num_episodes_train)
    logger.info('---------------------------------- STARTING EVALUATION --------------------------------------------------------------------------')
    agent.save(out_policy_file)
    for mode in eval_mode:
        logger.info("-----------------------------Starting evaluation for {} action selection-------------------------".format(mode))
        agent.test(num_episodes=args.num_episodes_test, test_mode=mode)
    # write to csv test scalar metrics:
    for _, metric in agent.test_metrics.items():
        metric.write_to_csv()
    return agent


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-word_emb_size", type=int, default=8, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=24, help="dimension of the hidden state")
    parser.add_argument("-max_len", type=int, default=10, help="max episode length")
    # parser.add_argument("-num_training_steps", type=int, default=1000, help="number of training_steps")
    parser.add_argument("-num_episodes_train", type=int, default=300, help="number of episodes training")
    parser.add_argument("-num_episodes_test", type=int, default=50, help="number of episodes test")
    parser.add_argument("-data_path", type=str, required=True,
                        help="data folder containing questions embeddings and img features")
    parser.add_argument("-out_path", type=str, required=True, help="out folder")
    parser.add_argument('-logger_level', type=str, default="INFO", help="level of logger")
    parser.add_argument('-gamma', type=float, default=1., help="gamma")
    parser.add_argument('-log_interval', type=int, default=10, help="gamma")
    parser.add_argument('-reward', type=str, default="levenshtein_", help="type of reward function")
    parser.add_argument('-lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('-eps', type=float, default=1e-08, help='epsilon value for adam optimizer')
    parser.add_argument('-model', type=str, default="lstm_word", help="model")
    parser.add_argument('-truncate_mode', type=str, help="truncation mode") # arg that says now if are truncating the action space or not.
    parser.add_argument('-k_min', type=int, default=1, help="minimum of truncated action space size for sample_va truncation mode")
    parser.add_argument('-p_th', type=float, help="probability threshold for proba threshold truncation mode")
    parser.add_argument('-K_epochs', type=int, default=10, help="# epochs of training each update_timestep")
    parser.add_argument('-update_every', type=int, default=20, help="update_every episode/timestep")
    parser.add_argument('-entropy_coeff', type=float, default=0.01, help="entropy coeff")
    parser.add_argument('-eps_clip', type=float, default=0.02, help="eps clip")
    parser.add_argument('-grad_clip', type=float, help="value of gradient norm clipping")
    parser.add_argument('-lm_path', type=str, required=True,
                        help="if specified, the language model truncate the action space")
    parser.add_argument('-lm_sl', type=int, default=1, help="the language model is trained with sl")
    parser.add_argument('-policy_path', type=str, default=None,
                        help="if specified, pre-trained model of the policy")
    parser.add_argument('-pretrain', type=int, default=0, help="the agent use pretraining on the dataset")
    parser.add_argument('-debug', type=str, default="0,69000",
                        help="debug mode: train on the first debug images")
    parser.add_argument('-resume_training', type=str, help='folder path to resume training from saved saved checkpoint')
    parser.add_argument('-agent', type=str, default="PPO", help="RL agent")
    parser.add_argument('-conv_kernel', type=int, default=1, help="conv kernel")
    parser.add_argument('-stride', type=int, default=2, help="stride conv")
    parser.add_argument('-num_filters', type=int, default=3, help="filters for conv")
    parser.add_argument('-num_truncated', type=int, default=10, help="number of words from lm")
    parser.add_argument('-num_questions', type=int, default=10, help="number of questions for each image")
    parser.add_argument('-diff_reward', type=int, default=0, help="is reward differential")
    return parser


if __name__ == '__main__':
    # -data_path /Users/guillaumequispe/PycharmProjects/RL-NLP/data -out_path /Users/guillaumequispe/PycharmProjects/RL-NLP/output
    # -max_len 7 -logger_level DEBUG -num_episodes_train 4000 -log_interval 1 -reward "levenshtein_"
    # -model lstm_word -update_timestep 50 -K_epochs 10 -entropy_coeff 0.01 -eps_clip 0.02
    parser = get_parser()
    args = parser.parse_args()
    run(args)
