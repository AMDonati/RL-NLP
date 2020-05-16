# https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/beam_search.py
import torch
import numpy as np
from collections import namedtuple
import os
from data_provider.CLEVR_Dataset import CLEVR_Dataset
import Levenshtein as lv
from RL_toolbox.action_selection import select_action
from models.Policy_network import PolicyLSTM

State = namedtuple('State', ('text', 'img'))
Episode = namedtuple('Episode', ('img_idx', 'img_feats', 'GD_questions', 'dialog', 'rewards'))

def get_dummy_reward(next_state_text, ep_questions, EOS_idx):
    next_state_text = next_state_text[:, 1:]  # removing sos token.
    if next_state_text[:, -1] == EOS_idx:  # remove <EOS> token if needed.
        next_state_text = next_state_text[:, :-1]
    # trunc state:
    state_len = next_state_text.size(1)
    if state_len == 0:
        #print('no final reward...')
        return 0.
    else:
        # max_len = ep_questions.size(1)
        # if state_len < max_len:
        #     next_state_text = torch.cat([next_state_text, next_state_text.new_zeros(1, max_len - state_len)], dim=1)
        # assert max_len == next_state_text.size(1)
        ep_questions = ep_questions[:, :state_len]
        next_state_text = next_state_text.repeat(ep_questions.size(0), 1)
        mask_same_tokens = next_state_text == ep_questions
        reward = mask_same_tokens.sum(dim=-1).max().numpy()
        reward = reward / state_len
    return reward

#TODO: batchify this function.
def get_levenshtein_reward(next_state_text, ep_questions, dataset, EOS_idx):
    next_state_text = next_state_text[:, 1:]  # removing sos token.
    if next_state_text[:, -1] == EOS_idx:  # remove <EOS> token if needed.
        next_state_text = next_state_text[:, :-1]
    # trunc state:
    state_len = next_state_text.size(1)
    if state_len == 0:
        # print('no final reward...')
        return 0.
    else:
        # decode GD questions.
        ep_questions = ep_questions.data.numpy()
        ep_questions = [list(ep_questions[i, :10]) for i in range(ep_questions.shape[0])]
        decoded_questions = [dataset.idx2word(question, stop_at_end=True) for question in ep_questions]
        # decode state.
        dialog = next_state_text.view(-1).data.numpy()
        decoded_dialog = dataset.idx2word(list(dialog), stop_at_end=True)
        # lv distance.
        dist = [lv.distance(q, decoded_dialog) / max(len(q),len(decoded_dialog)) for q in decoded_questions]
        sim = [1-d for d in dist]

        return max(sim)


# function generate one episode. debugged.
# TODO: batchify this function.
def generate_one_episode(clevr_dataset, policy_network, special_tokens, device, max_len=None, reward_fn='lv', select='greedy', seed=None):
    if max_len is None:
        max_len = clevr_dataset.input_questions.size(1)  # max_length set-up to max length of questions dataset (or avg len?)
    max_len = 10  # FOR DEBUGGING.
    # sample initial state
    if seed is not None:
        np.random.seed(seed)
    img_idx = np.random.randint(0, len(clevr_dataset.img_idxs))
    img_idx = 0  # FOR DEBUGGING.
    ep_GD_questions = clevr_dataset.get_questions_from_img_idx(img_idx)  # shape (10, S-1) # used to compute the final reward of the episode.
    img_feats = clevr_dataset.get_feats_from_img_idx(img_idx)  # shape (1024, 14, 14)
    initial_state = State(torch.LongTensor([special_tokens.SOS_idx]).view(1, 1), img_feats.unsqueeze(0))

    state = initial_state
    done = False
    step = 0
    rewards, log_probs = [], []
    while not done:
        if select == 'sampling':
            action, log_prob = select_action(policy_network, state, device, mode='sampling')
        elif select == 'greedy':
            action, log_prob = select_action(policy_network, state, device, mode='greedy')
        # compute next state, done, reward from the action.
        next_state = State(torch.cat([state.text, action], dim=1), state.img)
        done = True if action.item() == special_tokens.EOS_idx or step == (max_len - 1) else False
        if done:
            # reward = get_dummy_reward(next_state_text=next_state.text,
            #                           ep_questions=ep_GD_questions,
            #                           EOS_idx=special_tokens.EOS_idx)
            reward = get_levenshtein_reward(next_state_text=next_state.text,
                                            ep_questions=ep_GD_questions,
                                            dataset=clevr_dataset,
                                            EOS_idx=special_tokens.EOS_idx)
        else:
            reward = 0
            step += 1
        rewards.append(reward)
        log_probs.append(log_prob)
        state = next_state

    episode = Episode(img_idx, img_feats.data.numpy(), ep_GD_questions.data.numpy(), state.text.squeeze().data.numpy(), rewards)
    return_ep = sum(rewards)
    returns = [return_ep] * (step + 1)

    if len(returns) < max_len:
        assert state.text[:, -1] == special_tokens.EOS_idx

    return log_probs, returns, episode

def generate_episodes_batch(clevr_dataset, policy_network, special_tokens, device, BATCH_SIZE, max_len=None, reward_fn='lv', select='greedy', seed=None):
    if max_len is None:
        max_len = clevr_dataset.input_questions.size(1)  # max_length set-up to max length of questions dataset (or avg len?)
    # sample initial state
    if seed is not None:
        np.random.seed(seed)
    img_idx = list(np.random.randint(0, len(clevr_dataset), size=BATCH_SIZE))
    ep_GD_questions = [clevr_dataset.get_questions_from_img_idx(i) for i in img_idx] #TODO: debug shape (0,45) for some idx.
    #  shape (10, S-1) # used to compute the final reward of the episode.
    ep_GD_questions = torch.stack(ep_GD_questions, dim=0)
    img_feats = [clevr_dataset.get_feats_from_img_idx(i) for i in img_idx]  # shape (1024, 14, 14)
    img_feats = torch.stack(img_feats, dim=0)
    initial_state = State(torch.LongTensor([special_tokens.SOS_idx]).view(1, 1).repeat(BATCH_SIZE, 1), img_feats)

    state = initial_state
    done = False
    step = 0
    rewards, log_probs = [], []
    while not done:
        if select == 'sampling':
            action, log_prob = select_action(policy_network, state, device, mode='sampling')
        elif select == 'greedy':
            action, log_prob = select_action(policy_network, state, device, mode='greedy') # action (bs, 1), # log_probs (bs)
        # compute next state, done, reward from the action.
        next_state = State(torch.cat([state.text, action], dim=1), state.img)
        done = True if action.item() == special_tokens.EOS_idx or step == (max_len - 1) else False #TODO: adapt this to the batch case.
        if done:
            # reward = get_dummy_reward(next_state_text=next_state.text,
            #                           ep_questions=ep_GD_questions,
            #                           EOS_idx=special_tokens.EOS_idx)
            reward_batch = []
            for i in range(BATCH_SIZE):
                reward = get_levenshtein_reward(next_state_text=next_state.text[i,:],
                                            ep_questions=ep_GD_questions[i,:,:],
                                            dataset=clevr_dataset,
                                            EOS_idx=special_tokens.EOS_idx)
                reward_batch.append(reward)
        else:
            reward_batch = [0.] * BATCH_SIZE
            step += 1
        rewards.append(reward_batch)
        log_probs.append(log_prob)
        state = next_state

    episode = Episode(img_idx, img_feats.data.numpy(), ep_GD_questions.data.numpy(), state.text.data.numpy(), rewards)
    return_ep = sum(rewards)
    returns = [return_ep] * (step + 1)

    if len(returns) < max_len:
        assert state.text[:, -1] == special_tokens.EOS_idx

    return log_probs, returns, episode


def padder_batch(batch):
    len_episodes = [len(l) for l in batch]
    max_len = max(len_episodes)
    batch_tensors = [torch.tensor(l, dtype=torch.float32, requires_grad=True).unsqueeze(-1) for l in
                     batch]  # tensors of shape (len_ep, 1)
    batch_tensors_padded = [torch.cat([t, t.new_zeros(max_len - len, t.size(-1))]) for (t, len) in
                            zip(batch_tensors, len_episodes)]
    batch = torch.stack(batch_tensors_padded, dim=0)
    return batch


def train_episodes_batch(log_probs_batch, returns_batch, optimizer):
    reinforce_loss = -log_probs_batch * returns_batch  # shape (bs, max_len, 1) # opposite of REINFORCE objective function to apply a gradient descent algo.
    reinforce_loss = reinforce_loss.squeeze(-1).sum(dim=1).mean(dim=0)  # sum over timesteps, mean over batches.
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()
    return reinforce_loss.item()

if __name__ == '__main__':
    # test of get_dummy_reward function.
    h5_questions_path = os.path.join("../../data", 'train_questions.h5')
    h5_feats_path = os.path.join("../../data", 'train_features.h5')
    vocab_path = os.path.join("../../data", 'vocab.json')
    clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                  h5_feats_path=h5_feats_path,
                                  vocab_path=vocab_path)
    sample_questions = clevr_dataset.get_questions_from_img_idx(0)
    temp_state_text = torch.LongTensor([1, 7, 86, 70, 70, 21, 54, 81, 51, 84, 86, 50, 38, 17, 2]).unsqueeze(0)
    temp_reward = get_dummy_reward(temp_state_text, sample_questions, 2)
    print('reward', temp_reward)

    # test of levenstein reward function.
    temp_rew_lv = get_levenshtein_reward(temp_state_text, sample_questions, clevr_dataset, 2)
    print('temp rew', temp_rew_lv)

if __name__ == '__main__':
    State = namedtuple('State', ('text', 'img'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 8
    h5_questions_path = os.path.join("../../data", 'train_questions.h5')
    h5_feats_path = os.path.join("../../data", 'train_features.h5')
    vocab_path = os.path.join("../../data", 'vocab.json')
    clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                  h5_feats_path=h5_feats_path,
                                  vocab_path=vocab_path,
                                  max_samples=21)
    SOS_idx = clevr_dataset.vocab_questions["<SOS>"]
    EOS_idx = clevr_dataset.vocab_questions["<EOS>"]
    Special_Tokens = namedtuple('Special_Tokens', ('SOS_idx', 'EOS_idx'))
    special_tokens = Special_Tokens(SOS_idx, EOS_idx)

    policy_network = PolicyLSTM(num_tokens=87,
                                word_emb_size=16,
                                emb_size=16 + 16 * 7 * 7,
                                hidden_size=32,
                                num_layers=1,
                                p_drop=0)

    # -------- test of generate_episode_batch ----------------------------------------------------------------------------------------
    log_probs, action, episode = generate_episodes_batch(clevr_dataset=clevr_dataset,
                                                         policy_network=policy_network,
                                                         special_tokens=special_tokens,
                                                         device=device,
                                                         BATCH_SIZE=BATCH_SIZE)