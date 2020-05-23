import argparse
import datetime
import logging
import os
import random

import pandas as pd
import torch
from torch.utils.tensorboard import SummaryWriter

from agent.reinforce import REINFORCE
from envs.clevr_env import ClevrEnv
from models.rl_basic import PolicyGRUWord, PolicyGRU
from utils.utils_train import create_logger, str2bool


def train(env, agent, writer, output_path="lm",log_interval=10, num_episodes=100, pretrain=False, reduced_vocab=False):
    #writer = SummaryWriter(log_dir=os.path.join(output_path, 'runs'))
    running_reward = 0
    for i_episode in range(num_episodes):
        state, ep_reward = env.reset(), 0
        #dict_tokens, reduced_vocab = env.get_reduced_action_space() if reduced_vocab else None, None
        ref_question = env.ref_questions[random.randint(0, len(env.ref_questions) - 1)]
        for t in range(0, env.max_len + 1):
            forced = ref_question[t] if pretrain else None
            action, log_probs, value = agent.select_action(state, forced=forced)
            state, (reward, _), done, _ = env.step(action)
            if pretrain:
                value = torch.tensor([-1.])
            agent.model.rewards.append(reward)
            agent.model.values.append(value)
            agent.model.saved_log_probs.append(log_probs)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        agent.finish_episode()
        if i_episode % log_interval == 0:
            logging.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            writer.add_text('episode_questions', '\n'.join(env.ref_questions_decoded))
            writer.add_scalar('train_running_return', running_reward, i_episode + 1)

        df = pd.DataFrame(agent.model.last_policy[-env.max_len:])
        # diff_df = df.diff(periods=5)
        diff_df = (df.iloc[-1] - df.iloc[0]).abs()
        top_words = diff_df.nlargest(4)
        logging.info("top words changed in the policy : {}".format(env.clevr_dataset.idx2word(top_words.index)))

    out_file= os.path.join(output_path, 'model.pth')
    #with open(out_file, 'wb') as f:
    #    torch.save(agent.model, f)
    torch.save(agent.model.state_dict(), out_file)
    return agent


def test(env, agent, writer,log_interval=1, num_episodes=10, reduced_vocab=False, pretrained_model=None):
    running_reward = 0
    for i_episode in range(num_episodes):
        state, ep_reward = env.reset(), 0
        #dict_tokens, reduced_vocab = env.get_reduced_action_space() if reduced_vocab else None, None
        for t in range(0, env.max_len + 1):
            action, log_probs, value = agent.select_action(state)
            state, (reward, _), done, _ = env.step(action)
            # if args.render:
            # env.render()
            agent.model.rewards.append(reward)
            agent.model.values.append(value)
            agent.model.saved_log_probs.append(log_probs)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        # agent.finish_episode()
        if i_episode % log_interval == 0:
            logging.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            writer.add_text('episode_questions', '\n'.join(env.ref_questions_decoded))
            writer.add_scalar('test_running_return', running_reward, i_episode + 1)


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
    parser.add_argument('-pretrain', type=str2bool, default=0, help="pretraining with rl")
    parser.add_argument('-reduced_vocab', type=str2bool, default=False, help="reducing vocab")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_path = os.path.join(args.out_path, "train", "{}".format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    out_file_log = os.path.join(output_path, 'RL_training_log.log')

    logger = create_logger(out_file_log, level=args.logger_level)

    writer = SummaryWriter(log_dir=os.path.join(output_path, 'runs'))

    # csv_out_file = os.path.join(output_path, 'train_history.csv')
    # model_path = os.path.join(output_path, 'model.pt')
    # logger = create_logger("train.log", level=args.logger_level)

    env = ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode="train")
    # debug_true_questions=[[7, 8, 10, 12, 14]]

    models = {"gru_word": PolicyGRUWord,
              "gru": PolicyGRU}

    with open(args.pretrained_path, 'rb') as f:
        pretrained_lm = torch.load(f, map_location=device).to(device)
    model = models[args.model](env.clevr_dataset.len_vocab, args.word_emb_size, args.hidden_size)

    agent = REINFORCE(model=model, gamma=args.gamma, lr=args.lr)

    train(env=env, agent=agent, log_interval=args.log_interval, num_episodes=args.num_episodes_train,
          pretrain=args.pretrain,
          reduced_vocab=args.reduced_vocab, writer=writer, output_path=output_path)
    logging.info("-" * 20)
    logging.info("TEST")
    logging.info("-" * 20)

    # using val set because no answer in test set -> bug
    env = ClevrEnv(args.data_path, args.max_len, reward_type=args.reward, mode="val")
    test(env=env, agent=agent, reduced_vocab=args.reduced_vocab, num_episodes=args.num_episodes_test, writer=writer)
