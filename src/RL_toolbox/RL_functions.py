# https://github.com/MaximumEntropy/Seq2Seq-PyTorch/blob/master/beam_search.py
import torch
import numpy as np
from collections import namedtuple
import os
from data_provider.CLEVR_Dataset import CLEVR_Dataset
import Levenshtein as lv
from models.Policy_network import PolicyLSTM
from RL_toolbox.reward import Levenshtein, get_dummy_reward

State = namedtuple('State', ('text', 'img'))
Episode = namedtuple('Episode', ('img_idx', 'img_feats', 'GD_questions', 'closest_question', 'dialog', 'rewards'))

def masked_softmax(vec, mask, dim=1, epsilon=1e-5):
    exps = torch.exp(vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True) + epsilon
    return (masked_exps / masked_sums)

def preprocess_final_state(state_text, dataset, EOS_idx):
    state_text = state_text[:, 1:]  # removing sos token.
    if state_text[:, -1] == EOS_idx:  # remove <EOS> token if needed.
        state_text = state_text[:, :-1]
    if state_text.size(1) == 0:
        return None
    else:
        # decode state.
        dialog = state_text.view(-1).data.numpy()
        decoded_dialog = dataset.idx2word(list(dialog), stop_at_end=True)
    return decoded_dialog

def generate_one_episode(env, policy_network, device, select='greedy'):
    state = env.reset()
    done = False
    rewards, log_probs, values = [], []
    while not done:
        if select == 'sampling':
            action, log_prob, value = select_action(policy_network, state, device, mode='sampling')
        elif select == 'greedy':
            action, log_prob, value = select_action(policy_network, state, device, mode='greedy')
        # compute next state, done, reward from the action.
        state, (reward, closest_question), done, _ = env.step(action)
        rewards.append(reward)
        log_probs.append(log_prob)
        values.append(values)

    episode = Episode(env.img_idx, env.img_feats.data.numpy(), env.ref_questions_decoded, closest_question, env.dialog, rewards)

    if len(rewards) < env.max_len:
        assert state.text[:, -1] == env.special_tokens.EOS_idx

    return log_probs, rewards, episode

# def generate_episodes_batch(clevr_dataset, policy_network, special_tokens, device, BATCH_SIZE, max_len=None, select='greedy', seed=None):
#     if max_len is None:
#         max_len = clevr_dataset.input_questions.size(1)  # max_length set-up to max length of questions dataset (or avg len?)
#     # sample initial state
#     if seed is not None:
#         np.random.seed(seed)
#     img_idx = list(np.random.randint(0, len(clevr_dataset), size=BATCH_SIZE))
#     ep_GD_questions = [clevr_dataset.get_questions_from_img_idx(i) for i in img_idx] #TODO: debug shape (0,45) for some idx.
#     #  shape (10, S-1) # used to compute the final reward of the episode.
#     ep_GD_questions = torch.stack(ep_GD_questions, dim=0)
#     img_feats = [clevr_dataset.get_feats_from_img_idx(i) for i in img_idx]  # shape (1024, 14, 14)
#     img_feats = torch.stack(img_feats, dim=0)
#     initial_state = State(torch.LongTensor([special_tokens.SOS_idx]).view(1, 1).repeat(BATCH_SIZE, 1), img_feats)
#
#     state = initial_state
#     done = False
#     step = 0
#     rewards, log_probs = [], []
#     while not done:
#         if select == 'sampling':
#             action, log_prob = select_action(policy_network, state, device, mode='sampling')
#         elif select == 'greedy':
#             action, log_prob = select_action(policy_network, state, device, mode='greedy') # action (bs, 1), # log_probs (bs)
#         # compute next state, done, reward from the action.
#         next_state = State(torch.cat([state.text, action], dim=1), state.img)
#         done = True if action.item() == special_tokens.EOS_idx or step == (max_len - 1) else False #TODO: adapt this to the batch case.
#         if done:
#             # reward = get_dummy_reward(next_state_text=next_state.text,
#             #                           ep_questions=ep_GD_questions,
#             #                           EOS_idx=special_tokens.EOS_idx)
#             reward_batch = []
#             for i in range(BATCH_SIZE):
#                 reward = get_levenshtein_reward(next_state_text=next_state.text[i,:],
#                                             ep_questions=ep_GD_questions[i,:,:],
#                                             dataset=clevr_dataset,
#                                             EOS_idx=special_tokens.EOS_idx)
#                 reward_batch.append(reward)
#         else:
#             reward_batch = [0.] * BATCH_SIZE
#             step += 1
#         rewards.append(reward_batch)
#         log_probs.append(log_prob)
#         state = next_state
#
#     episode = Episode(img_idx, img_feats.data.numpy(), ep_GD_questions.data.numpy(), state.text.data.numpy(), rewards)
#     return_ep = sum(rewards)
#     returns = [return_ep] * (step + 1)
#
#     if len(returns) < max_len:
#         assert state.text[:, -1] == special_tokens.EOS_idx
#
#     return log_probs, returns, episode


def padder_batch(batch):
    len_episodes = [len(l) for l in batch]
    max_len = max(len_episodes)
    batch_tensors = [torch.tensor(l, dtype=torch.float32, requires_grad=True).unsqueeze(-1) for l in
                     batch]  # tensors of shape (len_ep, 1)
    batch_tensors_padded = [torch.cat([t, t.new_zeros(max_len - len, t.size(-1))]) for (t, len) in
                            zip(batch_tensors, len_episodes)]
    batch = torch.stack(batch_tensors_padded, dim=0).squeeze(dim=-1)
    return batch


def train_episodes_batch(log_probs_batch, returns_batch, optimizer):
    reinforce_loss = -log_probs_batch * returns_batch  # shape (bs, max_len, 1) # opposite of REINFORCE objective function to apply a gradient descent algo.
    reinforce_loss = reinforce_loss.squeeze(-1).sum(dim=1).mean(dim=0)  # sum over timesteps, mean over batches.
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()
    return reinforce_loss.item()

if __name__ == '__main__':
    h5_questions_path = os.path.join("../../data", 'train_questions.h5')
    h5_feats_path = os.path.join("../../data", 'train_features.h5')
    vocab_path = os.path.join("../../data", 'vocab.json')
    clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                  h5_feats_path=h5_feats_path,
                                  vocab_path=vocab_path,
                                  max_samples=21)

    # ---- test of get dummy reward -----------------------------------------
    sample_questions = clevr_dataset.get_questions_from_img_idx(0)
    temp_state_text = torch.LongTensor([1, 7, 86, 70, 70, 21, 54, 81, 51, 84, 86, 50, 38, 17, 2]).unsqueeze(0)
    temp_reward = get_dummy_reward(temp_state_text, sample_questions, 2)
    print('reward', temp_reward)

    State = namedtuple('State', ('text', 'img'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    SOS_idx = clevr_dataset.vocab_questions["<SOS>"]
    EOS_idx = clevr_dataset.vocab_questions["<EOS>"]
    PAD_idx = clevr_dataset.vocab_questions["<PAD>"]
    Special_Tokens = namedtuple('Special_Tokens', ('SOS_idx', 'EOS_idx', 'PAD_idx'))
    special_tokens = Special_Tokens(SOS_idx, EOS_idx, PAD_idx)

    policy_network = PolicyLSTM(num_tokens=87,
                                word_emb_size=16,
                                emb_size=16 + 16 * 7 * 7,
                                hidden_size=32,
                                num_layers=1,
                                p_drop=0)

    # -------- test of generate_episode  ----------------------------------------------------------------------------------------
    log_probs, returns, episode = generate_one_episode(clevr_dataset=clevr_dataset,
                                                         policy_network=policy_network,
                                                         special_tokens=special_tokens,
                                                         device=device)
    print('log_probs', len(log_probs))
    print('return', returns[-1])
    print('dialog', episode.dialog)
    print('closest question', episode.closest_question)

    # ------- test of preprocess functions -------------------------------------------------------------------------------------------
    #temp_state = torch.tensor([[1,2]])
    temp_state = torch.tensor([[1,10, 20, 62,2]])
    temp_q = torch.tensor([[13,22,34,10,7,0,0,0,0,0],[67, 45, 63, 0, 0, 0, 0, 0, 0, 0]])
    temp_dialog = preprocess_final_state(state_text=temp_state, dataset=clevr_dataset, EOS_idx=special_tokens.EOS_idx)
    temp_questions = preprocess_ep_questions(ep_questions=temp_q, dataset=clevr_dataset, PAD_idx=PAD_idx, max_len=10)
    print('temp question', temp_questions[0])
    print('temp question', temp_questions[1])
    print('temp dialog', temp_dialog)
