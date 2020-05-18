import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from data_provider.CLEVR_Dataset import CLEVR_Dataset
import os
from collections import namedtuple
import numpy as np
from models.Policy_network import PolicyLSTM

def select_action(policy_network, state, device, mode='greedy'):
    bs = state.text.size(0)
    seq_len = state.text.size(1)
    policy_network.train()
    state.text.to(device)
    state.img.to(device)
    logits, _ = policy_network(state.text, state.img) # logits > shape (s*num_samples, num_tokens)
    logits = logits.view(bs, seq_len, -1)
    probas = F.softmax(logits, dim=-1) # (num samples, s, num_tokens)
    if mode == 'sampling':
        m = Categorical(probas[:, -1, :])  # multinomial distribution with weights = probas.
        action = m.sample()
    elif mode == 'greedy':
        _, action = probas[:,-1,:].max(dim=-1)
        action = action.squeeze(-1)
    log_prob = F.log_softmax(logits, dim=-1)[:,-1,action]
    return action.view(1, 1), log_prob


def select_action_batch(policy_network, state, device, mode='greedy'):
    bs = state.text.size(0)
    seq_len = state.text.size(1)
    policy_network.train()
    state.text.to(device)
    state.img.to(device)
    logits, _ = policy_network(state.text, state.img) # logits > shape (s*num_samples, num_tokens)
    logits = logits.view(bs, seq_len, -1)
    probas = F.softmax(logits, dim=-1) # (num samples, s, num_tokens)
    if mode == 'sampling':
        m = Categorical(probas[:, -1, :])  # multinomial distribution with weights = probas.
        action = m.sample() # shape bs.
    elif mode == 'greedy':
        _, action = probas[:,-1,:].max(dim=-1)
        action = action.squeeze(-1)
    log_prob = [F.log_softmax(logits, dim=-1)[i, -1, action[i]] for i in range(bs)]
    log_prob = torch.stack(log_prob, dim=0)
    return action.view(bs, 1), log_prob


if __name__ == '__main__':

    State = namedtuple('State', ('text', 'img'))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    BATCH_SIZE = 1
    h5_questions_path = os.path.join("../../data", 'train_questions.h5')
    h5_feats_path = os.path.join("../../data", 'train_features.h5')
    vocab_path = os.path.join("../../data", 'vocab.json')
    clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                  h5_feats_path=h5_feats_path,
                                  vocab_path=vocab_path,
                                  max_samples=21)
    img_idx = list(np.random.randint(0, len(clevr_dataset), size=BATCH_SIZE))
    ep_GD_questions = [clevr_dataset.get_questions_from_img_idx(i) for i in
                       img_idx]  # shape (10, S-1) # used to compute the final reward of the episode.
    img_feats = [clevr_dataset.get_feats_from_img_idx(i) for i in img_idx]  # shape (1024, 14, 14)

    initial_state = State(torch.LongTensor([2]).view(1, 1).repeat(BATCH_SIZE, 1), torch.stack(img_feats, dim=0))
    policy_network = PolicyLSTM(num_tokens=87,
                                word_emb_size=16,
                                emb_size=16 + 16 * 7 * 7,
                                hidden_size=32,
                                num_layers=1,
                                p_drop=0)

    # test of select_action_sampling function.
    action, log_probs = select_action(policy_network=policy_network,
                                      state=initial_state,
                                      device=device,
                                      mode='sampling')
    print('action', action.shape)
    print('log_probs', log_probs.shape)

    action, log_probs = select_action(policy_network=policy_network,
                                               state=initial_state,
                                               device=device)
    print('action', action.shape)
    print('log_probs', log_probs.shape)