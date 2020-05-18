import argparse
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from data_provider.CLEVR_Dataset import CLEVR_Dataset
from envs.clevr_env import ClevrEnv

parser = argparse.ArgumentParser()
parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
parser.add_argument("-word_emb_size", type=int, default=12, help="dimension of the embedding layer")
parser.add_argument("-hidden_size", type=int, default=24, help="dimension of the hidden state")
parser.add_argument("-p_drop", type=float, default=0, help="dropout rate")
parser.add_argument("-grad_clip", type=float)
parser.add_argument("-lr", type=float, default=0.001)
parser.add_argument("-bs", type=int, default=16, help="batch size")
parser.add_argument("-max_len", type=int, default=10, help="max episode length")
parser.add_argument("-num_training_steps", type=int, default=1000, help="number of training_steps")
parser.add_argument("-data_path", type=str, required=True,
                    help="data folder containing questions embeddings and img features")
parser.add_argument("-out_path", type=str, required=True, help="out folder")
parser.add_argument('-num_workers', type=int, default=0, help="num workers for DataLoader")
parser.add_argument('-logger_level', type=str, default="INFO", help="level of logger")
parser.add_argument('-gamma', type=float, default=1., help="gamma")
parser.add_argument('-log_interval', type=int, default=10, help="gamma")
parser.add_argument('-reward', type=str, default="cosine", help="type of reward function")

args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

env = ClevrEnv(args.data_path, args.max_len, reward_type=args.reward)


# env.seed(args.seed)
# torch.manual_seed(args.seed)

class PolicyGRU(nn.Module):
    def __init__(self, num_tokens, word_emb_size, emb_size, hidden_size, num_filters=None, num_layers=1, p_drop=0,
                 pooling=True, cat_size=64 + 7 * 7 * 32):
        super(PolicyGRU, self).__init__()
        self.num_tokens = num_tokens
        self.emb_size = emb_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.p_drop = p_drop
        if num_filters is None:
            self.num_filters = word_emb_size
        else:
            self.num_filters = num_filters
        self.pooling = pooling
        self.dropout = nn.Dropout(p_drop)
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.gru = nn.GRU(word_emb_size, self.hidden_size, batch_first=True)
        self.fc = nn.Linear(cat_size, num_tokens)
        self.saved_log_probs = []
        self.rewards = []

        self.conv = nn.Conv2d(in_channels=1024, out_channels=self.num_filters, kernel_size=1)
        if pooling:
            self.max_pool = nn.MaxPool2d(kernel_size=2)
        self.last_policy = []

    def forward(self, text_inputs, img_feat):
        '''
        :param text_inputs: shape (S, B)
        :param img_feat: shape (B, C, H, W)
        :param hidden: shape (num_layers, B, hidden_size)
        :return:
        log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
        '''
        embed_text = self._get_embed_text(text_inputs)

        img_feat = F.relu(self.conv(img_feat))
        if self.pooling:
            img_feat = self.max_pool(img_feat)

        img_feat = img_feat.view(img_feat.size(0), -1)

        embedding = torch.cat((img_feat, embed_text.view(embed_text.size(0), -1)), dim=1)
        logits = self.fc(embedding)  # (S,B,num_tokens)
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        probs = F.softmax(logits, dim=1)
        # sumprobs = probs.sum().detach().numpy()
        # if math.isnan(sumprobs):
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        self.last_policy.append(probs.detach().numpy()[0])
        return policy_dist

    def _get_embed_text(self, text):
        _, hidden = self.gru(self.word_embedding(text))
        return hidden[-1]


class PolicyGRUWord(nn.Module):
    def __init__(self, num_tokens, word_emb_size, hidden_size, num_layers=1):
        super(PolicyGRUWord, self).__init__()
        self.num_tokens = num_tokens
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.word_embedding = nn.Embedding(num_tokens, word_emb_size)
        self.gru = nn.GRU(word_emb_size, self.hidden_size, batch_first=True)
        # self.fc = nn.Linear(hidden_size, num_tokens)
        self.fc = nn.Linear(hidden_size, num_tokens + 1)
        self.saved_log_probs = []
        self.rewards = []
        self.values = []
        self.last_policy = []

    def forward(self, text_inputs, img_feat):
        '''
        :param text_inputs: shape (S, B)
        :param img_feat: shape (B, C, H, W)
        :param hidden: shape (num_layers, B, hidden_size)
        :return:
        log_probas: shape (S*B, num_tokens), hidden (num_layers, B, hidden_size)
        '''
        embed_text = self._get_embed_text(text_inputs)
        out = self.fc(embed_text)  # (S,B,num_tokens)
        logits, value = out[:, :self.num_tokens], out[:, self.num_tokens]
        logits = logits.view(-1, self.num_tokens)  # (S*B, num_tokens)
        probs = F.softmax(logits, dim=1)
        policy_dist = Categorical(probs)
        probs = policy_dist.probs.clone()
        self.last_policy.append(probs.detach().numpy()[0])
        return policy_dist, value

    def _get_embed_text(self, text):
        _, hidden = self.gru(self.word_embedding(text))
        return hidden[-1]


h5_questions_path = os.path.join(args.data_path, 'train_questions.h5')
h5_feats_path = os.path.join(args.data_path, 'train_features.h5')
vocab_path = os.path.join(args.data_path, 'vocab.json')
clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                              h5_feats_path=h5_feats_path,
                              vocab_path=vocab_path)
num_tokens = clevr_dataset.len_vocab
policy = PolicyGRUWord(num_tokens=num_tokens,
                       word_emb_size=8,
                       hidden_size=16,
                       )
optimizer = optim.Adam(policy.parameters(), lr=1e-2)
eps = np.finfo(np.float32).eps.item()
epsilon = 1.


def select_action(state):
    # state = torch.from_numpy(state).float().unsqueeze(0)
    # probs = policy(state.text, state.img)
    # m = Categorical(probs[-1, :])
    m, value = policy(state.text, state.img)
    action = m.sample()
    #global epsilon
    #rand = np.random.random()
    #if rand < epsilon:
    #    print("random")
    #    action = torch.randint(0, num_tokens, (1,))
   # epsilon *= 0.999
     #print("epsilon {}".format(epsilon))

    # policy.saved_log_probs.append(m.log_prob(action).view(1))
    return action.item(), m.log_prob(action).view(1), value


def finish_episode():
    # Print model's state_dict
    # print("Model's state_dict:")
    states = policy.state_dict()
    # for param_tensor in policy.state_dict():
    # print(param_tensor, "\t", policy.state_dict()[param_tensor].size())
    states_optim = optimizer.state_dict()

    R = 0
    policy_loss = []
    returns = []
    mse = nn.MSELoss()
    for r in policy.rewards[::-1]:
        R = r + args.gamma * R
        returns.insert(0, R)
    returns = torch.tensor(returns)
    std = returns.std() + eps if len(returns) > 1 else 1
    # returns = (returns - returns.mean()) / std
    for log_prob, R, value in zip(policy.saved_log_probs, returns, policy.values):
        policy_loss.append(-log_prob * (R - value))
        ms = mse(value, R)
        policy_loss.append(ms.view(1))
    # print(policy.last_policy[-1])
    optimizer.zero_grad()
    policy_loss = torch.cat(policy_loss).sum()
    policy_loss.backward()
    # torch.nn.utils.clip_grad_norm_(policy.parameters(), 5)
    optimizer.step()
    del policy.rewards[:]
    del policy.saved_log_probs[:]
    del policy.values[:]


def main(num_episodes=100):
    running_reward = 0
    for i_episode in range(num_episodes):
        state, ep_reward = env.reset(), 0
        for t in range(0, env.max_len):
            action, log_probs, value = select_action(state)
            state, reward, done, _ = env.step(action)
            # if args.render:
            # env.render()
            policy.rewards.append(reward)
            policy.values.append(value)
            policy.saved_log_probs.append(log_probs)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
        #if running_reward > 0.9:
        #    print("Solved! Running reward is now {} and "
        #          "the last episode runs to {} time steps!".format(running_reward, t))
        #   break

        df = pd.DataFrame(policy.last_policy[-args.max_len:])
        # diff_df=df.diff(periods=5)
        diff_df = (df.iloc[-1] - df.iloc[0]).abs()
        top_words = diff_df.nlargest(4)
        print("top words changed in the policy : {}".format(clevr_dataset.decode(top_words.index)))
        # best_token=diff_df_mean.apply(lambda s, n: s.nlargest(n).index, axis=1, n=2)
        # print("hs")


if __name__ == '__main__':
    main(args.num_training_steps)
