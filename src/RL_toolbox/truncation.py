import logging

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

from RL_toolbox.RL_functions import masked_softmax

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mask_truncature(valid_actions, logits, num_tokens=86):
    mask = torch.zeros(logits.size(0), num_tokens).to(device)
    mask[:, valid_actions] = 1
    probs_truncated = masked_softmax(logits.clone().detach(), mask)
    # check that the truncation is right.
    sum_probs_va = probs_truncated[:, valid_actions].sum(dim=-1)
    try:
        assert torch.all(
            torch.abs(sum_probs_va - torch.ones(sum_probs_va.size()).to(device)) < 1e-6), "ERROR IN TRUNCATION FUNCTION"
    except AssertionError:
        logging.error("ERROR IN TRUNCATION FUNCTION")
    policy_dist_truncated = Categorical(probs_truncated)
    return policy_dist_truncated


def mask_inf_truncature(valid_actions, logits, num_tokens=87):
    mask = (torch.ones(logits.size(0), num_tokens) * -1e32).to(device)
    mask[:, valid_actions] = logits[:, valid_actions].clone().detach()
    probs_truncated = F.softmax(mask, dim=-1)
    # check that the truncation is right.
    if probs_truncated[:, valid_actions].sum(dim=-1) - 1 > 1e-6:
        print("ERROR IN TRUNCATION FUNCTION")
    policy_dist_truncated = Categorical(probs_truncated)
    return policy_dist_truncated


class Truncation:
    def __init__(self, agent, pretrained_lm=None):
        self.language_model = pretrained_lm
        self.alpha_logits_lm = agent.alpha_logits_lm
        self.dataset = agent.env.clevr_dataset
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def get_valid_actions(self, state, truncation):
        if not truncation:
            return None, None, 0, None, None
        with torch.no_grad():
            log_probas_lm, logits_lm, origin_log_probs_lm = self.language_model.forward(state.text.to(self.device))
            valid_actions, action_probs = self.truncate(log_probas_lm, logits_lm)
            return valid_actions, action_probs, logits_lm, log_probas_lm, origin_log_probs_lm

    def truncate(self, log_probas, logits):
        return None, None


class NoTruncation(Truncation):
    def __init__(self, agent, **kwargs):
        Truncation.__init__(self, agent, pretrained_lm=kwargs["pretrained_lm"])

    def get_valid_actions(self, state, truncation):
        if self.alpha_logits_lm > 0:
            with torch.no_grad():
                log_probas_lm, logits_lm, origin_log_probs_lm = self.language_model.forward(state.text.to(self.device))
        else:
            logits_lm, log_probas_lm, origin_log_probs_lm = 0, None, None
        return None, None, logits_lm, log_probas_lm, origin_log_probs_lm


class TopK(Truncation):
    def __init__(self, agent, **kwargs):
        Truncation.__init__(self, agent, pretrained_lm=kwargs["pretrained_lm"])
        self.num_truncated = kwargs["num_truncated"]

    def truncate(self, log_probas, logits):
        top_k_weights, top_k_indices = torch.topk(log_probas, self.num_truncated, sorted=True)
        return top_k_indices, top_k_weights.exp()


class ProbaThreshold(Truncation):
    '''See OverLeaf for details on this truncation fn.'''

    def __init__(self, agent, **kwargs):
        Truncation.__init__(self, agent, pretrained_lm=kwargs["pretrained_lm"])
        self.p_th = kwargs["p_th"]

    def truncate(self, log_probas, logits):
        probas = F.softmax(log_probas, dim=-1)
        probas_mask = torch.ge(probas, self.p_th)
        valid_actions = torch.nonzero(probas_mask, as_tuple=False)[:, 1]  # slice trick to get only the indices.
        action_probs = probas[:, valid_actions]
        assert torch.all(action_probs >= self.p_th), "ERROR in proba threshold truncation function"
        return valid_actions.unsqueeze(0), action_probs


class SampleVA(Truncation):
    def __init__(self, agent, **kwargs):
        '''See Overleaf for details on this truncation fn.'''
        Truncation.__init__(self, agent, pretrained_lm=kwargs["pretrained_lm"])
        self.k_max = kwargs["num_truncated"]

    def truncate(self, log_probas, logits):
        probas = F.softmax(log_probas, dim=-1)
        dist = Categorical(probas)
        actions = dist.sample(sample_shape=[self.k_max])
        valid_actions = torch.unique(actions)
        action_probs = dist.probs[:, valid_actions]
        return valid_actions.unsqueeze(0), action_probs


class TopP(Truncation):
    def __init__(self, agent, **kwargs):
        '''See Overleaf for details on this truncation fn.'''
        Truncation.__init__(self, agent, pretrained_lm=kwargs["pretrained_lm"])
        self.top_p = kwargs["top_p"]
        self.filter_value = -float("Inf")
        self.min_tokens_to_keep = 1

    def truncate(self, log_probas, logits):
        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
        # Remove tokens with cumulative probability above the threshold (token with 0 are kept)
        sorted_indices_to_remove = cumulative_probs > self.top_p
        # scatter sorted tensors to original indexing
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        _, valid_actions = torch.where(indices_to_remove == False)
        action_probs = F.softmax(logits[:, valid_actions], dim=-1)
        valid_actions = valid_actions.unsqueeze(dim=0)
        return valid_actions, action_probs


truncations = {"no_trunc": NoTruncation, "top_k": TopK, "proba_thr": ProbaThreshold, "sample_va": SampleVA,
               "top_p": TopP}

if __name__ == '__main__':
    temp = torch.tensor([0.1, 0.2, 0.5, 0.7, 0.9])
    temp_mask = temp.ge_(0.5)
    print(temp_mask.size())
    temp_logits = torch.tensor([10, 30, -10, -40, 50], dtype=torch.float32)
    temp_dist = Categorical(logits=temp_logits)
    temp_VA = temp_dist.sample([3])
    print(temp_VA)

    ###############
    # for unit tests.
    from agent.ppo import PPO
    from models.rl_basic import PolicyLSTMBatch
    from envs.clevr_env import ClevrEnv
    from torch.utils.tensorboard import SummaryWriter

    data_path = "../../data"
    out_path = "../../output/temp"
    writer = SummaryWriter(out_path)
    env = ClevrEnv(data_path, max_len=10, reward_type="levenshtein_", mode="train", debug="0,1",
                   num_questions=1, diff_reward=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyLSTMBatch(env.clevr_dataset.len_vocab, 32, 64)
    pretrained_lm = torch.load("../../output/best_model/model.pt", map_location=torch.device('cpu'))
    pretrained_lm.eval()
    agent = PPO(policy=policy, env=env, test_envs=[], out_path=out_path, writer=writer, pretrained_lm=pretrained_lm)
    state = env.reset()

    # test Top k
    print("top_k...")
    top_k = TopK(agent=agent, num_truncated=10)
    valid_actions, action_probs = top_k.get_valid_actions(state)
    print('valid_actions', valid_actions)
    print('valid_actions shape', valid_actions.size())
    print("action_probs", action_probs)
    print("action_probs", action_probs.size())
    print('valid actions decoded:', env.clevr_dataset.idx2word(valid_actions.squeeze().cpu().numpy()))
    dist, dist_truncated, val = top_k.get_policy_distributions(state, valid_actions)
    print('action sampled from policy dist truncated', dist_truncated.sample())
    action = top_k.sample_action(dist, dist_truncated, valid_actions)
    print("action", action)
    print("action shape", action.size())

    print("proba threshold...")
    # test proba threshold
    proba_thr = ProbaThreshold(agent=agent, p_th=0.01)
    valid_actions, action_probs = proba_thr.get_valid_actions(state)
    print('valid_actions:', valid_actions)
    print('action probs', action_probs)
    print("action_probs", action_probs.size())
    print('valid actions decoded:', env.clevr_dataset.idx2word(valid_actions.squeeze().cpu().numpy()))
    dist, dist_truncated, val = proba_thr.get_policy_distributions(state, valid_actions)
    print('action sampled from policy dist truncated', dist_truncated.sample())
    action = proba_thr.sample_action(dist, dist_truncated, valid_actions)
    print("action", action)
    print("action shape", action.size())

    print("sample_va...")
    # test sample_va:
    sample_va = SampleVA(agent=agent, num_truncated=10)
    valid_actions, action_probs = sample_va.get_valid_actions(state)
    print('valid_actions:', valid_actions)
    print('valid_actions shape', valid_actions.size())
    print('action probs', action_probs)
    print("action_probs shape", action_probs.size())
    print('valid actions decoded:', env.clevr_dataset.idx2word(valid_actions.squeeze().cpu().numpy()))
    dist, dist_truncated, val = sample_va.get_policy_distributions(state, valid_actions)
    print('action sampled from policy dist truncated', dist_truncated.sample())
    # test action sample from policy dist:
    action = sample_va.sample_action(dist, dist_truncated, valid_actions)
    print("action", action)
    print("action shape", action.size())

    print("top p ...")
    # test top p:
    top_p = TopP(agent=agent, top_p=0.5)
    valid_actions, action_probs = top_p.get_valid_actions(state)
