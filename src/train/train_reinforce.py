import argparse
import os

import torch

from agent.reinforce import REINFORCE
from envs.clevr_env import ClevrEnv


def train(env, agent, log_interval=10, num_episodes=100):
    running_reward = 0
    for i_episode in range(num_episodes):
        state, ep_reward = env.reset(), 0
        dict_tokens, reduced_vocab = env.get_reduced_action_space()
        for t in range(0, env.max_len + 1):
            action, log_probs, value = agent.select_action(state, valid_actions=dict_tokens)
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
        agent.finish_episode()
        if i_episode % log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        # df = pd.DataFrame(agent.model.last_policy[-max_len:])
        # diff_df=df.diff(periods=5)
        # diff_df = (df.iloc[-1] - df.iloc[0]).abs()
        # top_words = diff_df.nlargest(4)
        # print("top words changed in the policy : {}".format(env.clevr_dataset.idx2word(top_words.index)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-word_emb_size", type=int, default=12, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=24, help="dimension of the hidden state")
    parser.add_argument("-max_len", type=int, default=10, help="max episode length")
    parser.add_argument("-num_training_steps", type=int, default=1000, help="number of training_steps")
    parser.add_argument("-num_episodes", type=int, default=2000, help="number of episodes")

    parser.add_argument("-data_path", type=str, required=True,
                        help="data folder containing questions embeddings and img features")
    parser.add_argument("-out_path", type=str, required=True, help="out folder")
    parser.add_argument('-logger_level', type=str, default="INFO", help="level of logger")
    parser.add_argument('-gamma', type=float, default=1., help="gamma")
    parser.add_argument('-log_interval', type=int, default=10, help="gamma")
    parser.add_argument('-reward', type=str, default="levenshtein", help="type of reward function")
    parser.add_argument('-lr', type=float, default=0.005, help="learning rate")
    parser.add_argument('-debug_len_vocab', type=int, default=None, help="learning rate")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    h5_questions_path = os.path.join(args.data_path, 'train_questions.h5')
    h5_feats_path = os.path.join(args.data_path, 'train_features.h5')
    vocab_path = os.path.join(args.data_path, 'vocab.json')

    env = ClevrEnv(args.data_path, args.max_len, reward_type=args.reward)
    # debug_true_questions=[[7, 8, 10, 12, 14]]

    agent = REINFORCE(args.hidden_size, args.word_emb_size, env.clevr_dataset.len_vocab, gamma=args.gamma, lr=args.lr)

    train(env=env, agent=agent, log_interval=args.log_interval, num_episodes=args.num_episodes)
