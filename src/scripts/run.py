import argparse
import datetime
import os
from collections import OrderedDict
from configparser import ConfigParser

import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoModelWithLMHead, AutoTokenizer

from agent.ppo import PPO
from agent.reinforce import REINFORCE
from envs.clevr_env import ClevrEnv, VQAEnv
from models.language_model import GenericLanguageModel, ClevrLanguageModel
from models.rl_basic import PolicyLSTMBatch
from utils.utils_train import create_logger
from torch import optim
from torch.optim import lr_scheduler
import sys


def get_agent(pretrained_lm, writer, output_path, env, test_envs, policy, optimizer, args_):
    generic_kwargs = {"pretrained_lm": pretrained_lm,
                      "optimizer": optimizer,
                      "pretrain": args_.pretrain,
                      "update_every": args_.update_every,
                      "lr": args_.lr,
                      "grad_clip": args_.grad_clip, "writer": writer,
                      "truncate_mode": args_.truncate_mode,
                      "num_truncated": args_.num_truncated,
                      "p_th": args_.p_th,
                      "out_path": output_path,
                      "log_interval": args_.log_interval, "env": env,
                      "test_envs": test_envs,
                      "eval_no_trunc": args_.eval_no_trunc,
                      "alpha_logits": args_.alpha_logits,
                      "alpha_decay_rate": args_.alpha_decay_rate,
                      "epsilon_truncated": args_.epsilon_truncated,
                      "epsilon_truncated_rate": args_.epsilon_truncated_rate,
                      "train_seed": args_.train_seed,
                      "is_loss_correction": args_.is_loss_correction,
                      "train_metrics": args_.train_metrics,
                      "test_metrics": args_.test_metrics,
                      "top_p": args_.top_p,
                      "temperature": args_.temperature,
                      "temperature_step": args_.temp_step,
                      "temp_factor": args_.temp_factor,
                      "temperature_min": args_.temp_min,
                      "temperature_max": args_.temp_max,
                      "s_min": args_.s_min,
                      "s_max": args_.s_max,
                      "inv_schedule_step": args_.inv_schedule_step,
                      "schedule_start": args_.schedule_start,
                      "curriculum": args_.curriculum,
                      "KL_coeff": args_.KL_coeff,
                      "truncation_optim": args_.truncation_optim}

    ppo_kwargs = {"policy": policy, "gamma": args_.gamma,
                  "K_epochs": args_.K_epochs,
                  "entropy_coeff": args_.entropy_coeff,
                  "eps_clip": args_.eps_clip}
    reinforce_kwargs = {"policy": policy, "gamma": args_.gamma}
    algo_kwargs = {"PPO": ppo_kwargs, "REINFORCE": reinforce_kwargs}
    kwargs = {**algo_kwargs[args_.agent], **generic_kwargs}

    agents = {"PPO": PPO, "REINFORCE": REINFORCE}

    agent = agents[args_.agent](**kwargs)
    return agent


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str,
                        help="data folder containing questions embeddings and img features")
    parser.add_argument("-features_path", type=str,
                        help="data folder containing img features (VQA task)", default="data/vqa-v2/coco_trainval.lmdb")
    parser.add_argument("-out_path", type=str, help="out folder")
    # model args
    parser.add_argument('-model', type=str, default="lstm", help="model")
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-word_emb_size", type=int, default=32, help="dimension of the embedding layer")
    parser.add_argument("-attention_dim", type=int, default=64, help="dimension of the attention")
    parser.add_argument("-hidden_size", type=int, default=64, help="dimension of the hidden state")
    parser.add_argument('-conv_kernel', type=int, default=1, help="conv kernel")
    parser.add_argument('-stride', type=int, default=2, help="stride conv")
    parser.add_argument('-num_filters', type=int, default=3, help="filters for conv")
    parser.add_argument('-fusion', type=str, default="cat", help="fusion mode")
    # RL algo args.
    parser.add_argument('-agent', type=str, default="PPO", help="RL agent")
    parser.add_argument('-K_epochs', type=int, default=20, help="# epochs of training each update_timestep")
    parser.add_argument('-update_every', type=int, default=20, help="update_every episode/timestep")
    parser.add_argument('-entropy_coeff', type=float, default=0.01, help="entropy coeff")
    parser.add_argument('-eps_clip', type=float, default=0.02, help="eps clip")
    parser.add_argument('-optimizer', type=str, default="adam")
    parser.add_argument('-opt_schedule', type=str)
    parser.add_argument('-div_factor', type=int, default=25, help="div factor for OneCycleLR scheduler.")
    parser.add_argument('-lr', type=float, default=0.001, help="learning rate")
    parser.add_argument('-grad_clip', type=float, help="value of gradient norm clipping")
    parser.add_argument('-policy_path', type=str, default=None,
                        help="if specified, pre-trained model of the policy")
    # RL task args.
    parser.add_argument("-env", type=str, default="clevr", help="choice of the RL Task. Possible values: clevr or vqa.")
    parser.add_argument("-max_len", type=int, default=10, help="max episode length")
    parser.add_argument('-gamma', type=float, default=1., help="gamma")
    parser.add_argument('-reward', type=str, default="lv_norm", help="type of reward function")
    parser.add_argument('-reward_path', type=str, help="path for the reward")
    parser.add_argument('-reward_vocab', type=str, help="vocab for the reward")
    parser.add_argument("-params_reward", type=int, default=10, help="params reward")
    parser.add_argument('-mask_answers', type=int, default=1, help="mask answers")
    parser.add_argument('-answer_sampl', type=str, default="img_sampling",
                        help="method to sample the (img, answer) sample in the RL training.")
    parser.add_argument('-curriculum', type=int, default=0,
                        help="if > 0, changing the answer sampling mode from random to uniform")
    parser.add_argument('-debug', type=str, default="0,69000",
                        help="debug mode: train on the first debug images")
    parser.add_argument('-num_questions', type=int, default=10, help="number of questions for each image")
    parser.add_argument('-diff_reward', type=int, default=0, help="is reward differential")
    parser.add_argument('-condition_answer', type=str, default="none",
                        help="type of answer condition, default to none")
    parser.add_argument("-min_data", type=int, default=0)
    # truncation args.
    ## truncation mode args.
    parser.add_argument('-lm_path', type=str, default="gpt",
                        help="the language model path (used for truncating the action space if truncate_mode is not None).Else, used only at test time")
    parser.add_argument('-truncate_mode', type=str,
                        help="truncation mode")  # arg that says now if are truncating the action space or not.
    parser.add_argument('-num_truncated', type=int, default=10, help="number of words from lm")
    parser.add_argument('-p_th', type=float,
                        help="probability threshold for proba threshold truncation mode")  # arg used in the proba_thr truncation function.
    parser.add_argument('-top_p', default=1., type=float, help="top p of nucleus sampling")
    parser.add_argument('-s_min', default=1, type=int,
                        help="minimal size of the valid action space of the truncation function.")
    parser.add_argument('-s_max', default=200, type=int, help="maximal size of the valid action space")
    parser.add_argument('-KL_coeff', default=0., type=float, help="adding KL divergence term in the loss if truncation")
    ## temperature args.
    parser.add_argument('-temperature', default=1., type=float, help="temperature for language model")
    parser.add_argument('-temp_step', type=int, default=1,
                        help="temperature step for updating the temperature for the language model")
    parser.add_argument('-temp_factor', type=float, default=1., help="temperature factor for the language model")
    parser.add_argument('-temp_min', type=float, default=1., help="temperature min for the language model")
    parser.add_argument('-temp_max', type=float, default=10., help="temperature max for the language model")
    parser.add_argument('-inv_schedule_step', type=int, default=0, help="step to inverse the temperature schedule.")
    parser.add_argument('-schedule_start', type=int, default=1, help="step to start the temperature scheduling.")
    ## alpha logits fusion args.
    parser.add_argument('-alpha_logits', default=0., type=float,
                        help="alpha value for the convex logits mixture. if 0, does not fuse the logits of the policy with the logits of the lm.")
    parser.add_argument('-alpha_decay_rate', default=0., type=float,
                        help="alpha decay rate for the convex logits mixture. if 0, does not decay the alpha")
    ## epsilon truncation args.
    parser.add_argument('-epsilon_truncated', type=float, default=0.,
                        help="the agent sample from truncated or total action space")
    parser.add_argument('-epsilon_truncated_rate', type=float, default=1,
                        help="number of training iterations before epsilon truncated set to 1")
    parser.add_argument('-is_loss_correction', type=int, default=1,
                        help="adding the importance sampling ratio correction in the rl loss.")
    # gpt-2 pre-conditioning args.
    parser.add_argument('-init_text', type=str)
    parser.add_argument('-custom_init', type=int, default=0)
    parser.add_argument('-add_answers', type=int, default=0)
    # train / test pipeline:
    parser.add_argument("-num_episodes_train", type=int, default=10, help="number of episodes training")
    parser.add_argument("-num_episodes_test", type=int, default=10, help="number of episodes test")
    parser.add_argument("-train_seed", type=int, default=0,
                        help="using a seed for the episode generation in training or not...")
    parser.add_argument("-test_seed", type=int, default=1,
                        help="using a seed for the episode generation in test or not...")
    parser.add_argument('-resume_training', type=str, help='folder path to resume training from saved checkpoint')
    parser.add_argument('-eval_no_trunc', type=int, default=1,
                        help="if using truncation at training: at test time, evaluate also langage generated without truncation. Default to False.")
    parser.add_argument('-train_metrics', nargs='+', type=str,
                        default=["return", "size_valid_actions", "ppl_dialog_lm",
                                 "valid_actions", "dialog",
                                 "histogram_answers",
                                 "ttr", "sum_probs",
                                 "dialogimage"], help="train metrics")
    parser.add_argument('-test_metrics', nargs='+', type=str,
                        default=["return",],
                        help="test metrics")
    parser.add_argument('-test_modes', nargs='+', type=str,
                        default=["test_images"],
                        help="test metrics")
    parser.add_argument('-eval_modes', nargs='+', type=str,
                        default=['sampling', 'greedy', 'sampling_ranking_lm'],
                        help="test metrics")
    # misc.
    parser.add_argument('-logger_level', type=str, default="INFO", help="level of logger")
    parser.add_argument('-log_interval', type=int, default=10, help="log interval")
    parser.add_argument('-pretrain', type=int, default=0, help="the agent use pretraining on the dataset")
    parser.add_argument('-device_id', type=int, default=0, help="device id when running on a multi-GPU VM.")
    parser.add_argument('-num_diversity', type=int, default=1,
                        help="number of sampling for the same image/answer for test")
    parser.add_argument('-reduced_answers', type=int, default=0, help="reduced answers")
    parser.add_argument('-truncation_optim', type=int, default=0,
                        help="optimize the truncated distribution instead of the full one")
    parser.add_argument('-filter_numbers', type=int, default=0)

    return parser


def create_cmd_file(cmd_file_path):
    cmd_file = open(cmd_file_path, 'w')
    cmd_file.write(" ".join(sys.argv))
    cmd_file.close()


def create_config_file(conf_file, args):
    config = ConfigParser()
    config.add_section('main')
    for key, value in vars(args).items():
        config.set('main', key, str(value))
    with open(conf_file, 'w') as fp:
        config.write(fp)


def get_pretrained_lm(args, env, device):
    if "gpt" == args.lm_path:
        lm_model = AutoModelWithLMHead.from_pretrained("gpt2")
        tokenizer = AutoTokenizer.from_pretrained("gpt2")
        pretrained_lm = GenericLanguageModel(pretrained_lm=lm_model, dataset=env.dataset,
                                             tokenizer=tokenizer, init_text=args.init_text,
                                             custom_init=args.custom_init, add_answers=args.add_answers, device=device)
    else:
        lm_model = torch.load(args.lm_path, map_location=torch.device('cpu'))
        lm_model.eval()
        pretrained_lm = ClevrLanguageModel(pretrained_lm=lm_model, dataset=env.dataset,
                                           tokenizer=env.dataset.question_tokenizer, device=device)
    return pretrained_lm


def get_output_path(args):
    if args.truncate_mode is None:
        algo = "scratch" if args.policy_path is None else ""
    elif args.truncate_mode == 'top_k' or args.truncate_mode == 'sample_va':
        algo = "{}{}".format(args.truncate_mode, args.num_truncated)
    elif args.truncate_mode == "proba_thr":
        algo = "{}{}".format(args.truncate_mode, args.p_th)
    elif args.truncate_mode == "top_p":
        algo = "{}{}".format(args.truncate_mode, args.top_p)

    out_folder = '{}_{}_{}_answ-{}'.format(args.env, args.reward, args.agent, args.answer_sampl)
    if args.diff_reward:
        out_folder = out_folder + '_diffrew'

    if args.policy_path is not None:
        out_folder = out_folder + '_' + "pretrain"
    out_folder = out_folder + '_' + algo
    if float(args.epsilon_truncated) > 0:
        out_folder = out_folder + '_' + 'eps{}'.format(args.epsilon_truncated)
    if float(args.alpha_logits) != 0:
        out_folder = out_folder + '_' + 'alpha-logits{}'.format(args.alpha_logits)

    if "gpt" in args.lm_path:
        out_folder = out_folder + '_gpt-2'

    # optimization params
    out_folder = out_folder + '_{}_{}'.format(args.optimizer, args.lr)
    if args.opt_schedule is not None:
        out_folder = out_folder + '_{}{}'.format(args.opt_schedule, args.div_factor)
    out_folder = out_folder + '_ent{}'.format(args.entropy_coeff)
    out_folder = out_folder + '_epsclip{}'.format(args.eps_clip)
    out_folder = out_folder + '_graclip{}'.format(args.grad_clip)

    # temp args
    if args.temp_factor != 1:
        out_folder = out_folder + '_temp{}'.format(args.temperature) + '_div{}'.format(
            args.temp_factor) + '_step{}'.format(args.temp_step) + '_tmin{}'.format(args.temp_min) + '_tmax{}'.format(
            args.temp_max) + '_smin{}_smax{}'.format(args.s_min, args.s_max)
    if args.inv_schedule_step != 0:
        out_folder = out_folder + '_invsch{}'.format(args.inv_schedule_step)
    if args.schedule_start > 1:
        out_folder = out_folder + '_schstart{}'.format(args.schedule_start)

    if args.KL_coeff > 0:
        out_folder = out_folder + '_KLdiv{}'.format(args.KL_coeff)

    if args.resume_training is not None:
        output_path = os.path.join(args.resume_training,
                                   "resume_training_{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    else:
        if args.pretrain == 0:
            output_path = os.path.join(args.out_path, out_folder,
                                       "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
        else:
            output_path = os.path.join(args.out_path, out_folder, "pretrain",
                                       "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    stats_path = os.path.join(output_path, "stats")
    metric_path = os.path.join(output_path, "metrics")
    diversity_path = os.path.join(output_path, "diversity")

    if not os.path.isdir(stats_path):
        os.makedirs(stats_path)
    if not os.path.isdir(metric_path):
        os.makedirs(metric_path)
    if not os.path.isdir(diversity_path):
        os.makedirs(diversity_path)
    return output_path


def get_optimizer(policy, args):
    if args.optimizer == "adam":
        optimizer = optim.Adam(policy.parameters(), lr=args.lr)
    elif args.optimizer == "sgd":
        optimizer = optim.SGD(policy.parameters(), lr=args.lr)
    elif args.optimizer == "rmsprop":
        optimizer = optim.RMSprop(policy.parameters(), lr=args.lr)
    scheduler = args.opt_schedule
    if scheduler == "cyclic":
        scheduler = lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.div_factor * args.lr,
                                            total_steps=args.num_episodes_train)
    elif scheduler == "cyclic_multi":
        scheduler = lr_scheduler.CyclicLR(optimizer=optimizer, base_lr=args.lr, max_lr=args.div_factor * args.lr)
    elif scheduler == "WR":
        T_0 = max(1, int(args.num_episodes_train / 1000))
        scheduler = lr_scheduler.CosineAnnealingWarmRestarts(optimizer=optimizer, T_0=T_0)

    return optimizer, scheduler


def log_hparams(logger, args):
    logger.info('-' * 20 + 'Experience hparams' + '-' * 20)
    logger.info('TASK:{}'.format(args.env))
    logger.info('TASK REWARD: {}'.format(args.reward))
    logger.info("Episode Max Length: {}".format(args.max_len))
    logger.info("Number of Images: {}".format(args.debug.split(',')[1]))
    pre_trained = "scratch" if args.policy_path is None else "pretrain"
    logger.info("RL from {} ...".format(pre_trained))
    if args.truncate_mode is not None:
        logger.info("with truncation...")
        if "gpt" in args.lm_path:
            logger.info("with GPT-2 Language Model...")
        else:
            logger.info("with Dataset Language Model...")
        logger.info("Truncation mode: {}".format(args.truncate_mode))
        if args.truncate_mode == 'top_k' or args.truncate_mode == 'sample_va':
            logger.info("num_truncated:{}".format(args.num_truncated))
        elif args.truncate_mode == "proba_thr":
            logger.info("proba threshold:{}".format(args.p_th))
        elif args.truncate_mode == "top_p":
            logger.info("sum threshold:{}".format(args.top_p))
        if float(args.epsilon_truncated) > 0:
            logger.info("epsilon truncation with eps: {}".format(args.epsilon_truncated))
    else:
        logger.info("without truncation...")
    if float(args.alpha_logits) != 0:
        logger.info("logits fusion with alpha logits: {}".format(args.alpha_logits))
    logger.info("Number of TRAINING EPISODES: {}".format(args.num_episodes_train))
    logger.info("Number of TEST EPISODES: {}".format(args.num_episodes_test))


def get_rl_env(args, device):
    # upload env.
    if args.env == "clevr":
        env = ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode="train", debug=args.debug,
                       num_questions=args.num_questions, diff_reward=args.diff_reward, reward_path=args.reward_path,
                       reward_vocab=args.reward_vocab, mask_answers=args.mask_answers, device=device,
                       reduced_answers=args.reduced_answers, params=args.params_reward)
        test_modes = args.test_modes
        test_envs = [ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode=mode, debug=args.debug,
                              num_questions=args.num_questions, reward_path=args.reward_path,
                              reward_vocab=args.reward_vocab, mask_answers=args.mask_answers, device=device,
                              reduced_answers=args.reduced_answers, params=args.params_reward)
                     for mode in test_modes]
    elif args.env == "vqa":
        env = VQAEnv(args.data_path, features_h5path=args.features_path, max_len=args.max_len,
                     reward_type=args.reward, mode="train", max_seq_length=23, debug=args.debug,
                     diff_reward=args.diff_reward, reward_path=args.reward_path,
                     reward_vocab=args.reward_vocab, mask_answers=args.mask_answers, device=device,
                     min_data=args.min_data, reduced_answers=args.reduced_answers, answer_sampl=args.answer_sampl,
                     params=args.params_reward, filter_numbers=args.filter_numbers)
        if device.type == "cpu":
            test_envs = [env]
        else:
            test_envs = []
            if "test_images" in args.test_modes:
                test_envs.append(VQAEnv(args.data_path, features_h5path=args.features_path, max_len=args.max_len,
                                        reward_type=args.reward, mode="test_images", max_seq_length=23,
                                        debug=args.debug,
                                        diff_reward=args.diff_reward, reward_path=args.reward_path,
                                        reward_vocab=args.reward_vocab, mask_answers=args.mask_answers, device=device,
                                        min_data=args.min_data, reduced_answers=args.reduced_answers,
                                        answer_sampl="random", params=args.params_reward,
                                        filter_numbers=args.filter_numbers))
            if "test_text" in args.test_modes:
                test_text_env = env
                test_text_env.update_mode("test_text", answer_sampl="random")
                test_envs.append(test_text_env)

    return env, test_envs


def run(args):
    # check consistency hparams
    if args.reward == "vqa":
        assert args.condition_answer is not None, "VQA task should be conditioned on the answer"

    # create out_folder, config file, logger, writer
    output_path = get_output_path(args)
    conf_file = os.path.join(output_path, 'conf.ini')
    out_file_log = os.path.join(output_path, 'RL_training_log.log')
    out_policy_file = os.path.join(output_path, 'model.pth')
    cmd_file = os.path.join(output_path, 'cmd.txt')
    create_config_file(conf_file, args)
    create_cmd_file(cmd_file)
    logger = create_logger(out_file_log, level=args.logger_level)
    writer = SummaryWriter(log_dir=os.path.join(output_path, "runs"))

    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")

    # log hparams:
    log_hparams(logger, args)

    # upload env & pretrained lm, policy network.
    env, test_envs = get_rl_env(args, device)
    pretrained_lm = get_pretrained_lm(args, env, device)
    # dataset statistics
    logger.info('-' * 20 + 'Dataset statistics' + '-' * 20)
    logger.info("number of training questions:{}".format(len(env.dataset)))
    logger.info("vocab size:{}".format(len(env.dataset.vocab_questions)))

    models = {"lstm": PolicyLSTMBatch}
    # creating the policy model.
    policy = models[args.model](env.dataset.len_vocab, args.word_emb_size, args.hidden_size,
                                kernel_size=args.conv_kernel,
                                stride=args.stride, num_filters=args.num_filters,
                                fusion=args.fusion, env=env,
                                condition_answer=args.condition_answer,
                                device=device, attention_dim=args.attention_dim)
    if args.policy_path is not None:
        pretrained = torch.load(args.policy_path, map_location=device)
        if pretrained.__class__ != OrderedDict:
            if pretrained.__class__ == dict:
                pretrained = pretrained["model_state_dict"]
            else:
                pretrained = pretrained.state_dict()
        policy.load_state_dict(pretrained, strict=False)
        policy.device = device
    optimizer, scheduler = get_optimizer(policy, args)
    agent = get_agent(pretrained_lm=pretrained_lm, writer=writer, output_path=output_path, env=env, test_envs=test_envs,
                      policy=policy, optimizer=optimizer, args_=args)

    eval_mode = ['sampling', 'greedy', 'sampling_ranking_lm']

    # start training
    if args.resume_training is not None:
        epoch, loss = agent.load_ckpt(os.path.join(args.resume_training, "checkpoints"))
        logger.info('resume training after {} episodes... current loss: {}'.format(epoch, loss))
        agent.start_episode = epoch + 1
    if args.num_episodes_train > 0:  # trick to avoid a bug inside the agent.learn function in case of no training.
        agent.learn(num_episodes=args.num_episodes_train)
        agent.save(out_policy_file)
    else:
        logger.info("skipping training...")

    # start evaluation
    logger.info(
        '---------------------------------- STARTING EVALUATION --------------------------------------------------------------------------')
    for mode in args.eval_modes:
        logger.info(
            "----------------------------- Starting evaluation for {} action selection -------------------------".format(
                mode))
        agent.test(num_episodes=args.num_episodes_test, test_mode=mode, test_seed=args.test_seed)
    # write to csv test scalar metrics:
    agent.compute_write_all_metrics(output_path=output_path, logger=logger)
    logger.info(
        '------------------------------------ DONE ---------------------------------------------------------------')
    return agent


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    run(args)
