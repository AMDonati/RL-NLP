import numpy as np
import torch
import logging
from nltk.translate.bleu_score import sentence_bleu
import os
from utils.utils_train import write_to_csv

class Truncation:
    def __init__(self, agent, sample_action="dist_truncated"):
        self.agent = agent
        self.sample_action = sample_action

    def get_valid_actions(self, state_text, state_img, num_truncated=10):
        pass

    def get_policy_distributions(self):
        pass

    def sample_action(self):
        pass

class TopK(Truncation):
    def __init__(self, agent, sample_action="dist_truncated", num_truncated=10):
        Truncation.__init__(self, agent, sample_action)
        self.num_truncated = num_truncated

    def get_valid_actions(self, state_text, state_img, num_truncated=10):
        if self.agent.lm_sl:
            seq_len = state_text.size(1)
            log_probas, _ = self.agent.pretrained_lm(state_text.to(self.agent.device))
            log_probas = log_probas.view(len(state_text), seq_len, -1)
            log_probas = log_probas[:, -1, :]
            top_k_weights, top_k_indices = torch.topk(log_probas, self.num_truncated, sorted=True)
        else:
            dist, dist_, value = self.agent.pretrained_lm(state_text, state_img)
            probs = dist.probs
            top_k_weights, top_k_indices = torch.topk(probs, self.num_truncated, sorted=True)
        return top_k_indices, top_k_weights

    def get_policy_distributions(self, state, valid_actions):
        policy_dist, policy_dist_truncated, value = self.policy(state.text, state.img, valid_actions)
        #TODO: change here between self.policy_old and self.policy depending on PPO and REINFORCE.
        return policy_dist, policy_dist_truncated, value

    def sample_action(self, policy_dist, policy_dist_truncated, valid_actions):
        if self.sample_action == 'dist_truncated':
            action = policy_dist_truncated.sample()
            if policy_dist_truncated.probs.size() != policy_dist.probs.size():
                action = torch.gather(valid_actions, 1, action.view(1, 1))
        elif self.sample_action == 'dist': #TODO: debug here.
            action = policy_dist.sample()
            while action not in valid_actions:
                action = policy_dist.sample()
        return action

class PThreshold(Truncation):
    def __init__(self, agent, sample_action="dist_truncated", p_th = 0.01):
        Truncation.__init__(self, agent, sample_action)
        self.p_th = p_th

    def get_valid_actions(self, state_text, state_img, num_truncated=10):
        if self.agent.lm_sl:
            seq_len = state_text.size(1)
            log_probas, _ = self.agent.pretrained_lm(state_text.to(self.agent.device))
            log_probas = log_probas.view(len(state_text), seq_len, -1)
            probas = log_probas[:, -1, :].exp()
            #TODO: go on...
        else:
            dist, dist_, value = self.agent.pretrained_lm(state_text, state_img)
            probs = dist.probs
        return top_k_indices, top_k_weights

    def get_policy_distributions(self, state, valid_actions):
        policy_dist, policy_dist_truncated, value = self.policy(state.text, state.img, valid_actions)
        #TODO: change here between self.policy_old and self.policy depending on PPO and REINFORCE.
        return policy_dist, policy_dist_truncated, value

    def sample_action(self, policy_dist, policy_dist_truncated, valid_actions):
        if self.sample_action == 'dist_truncated':
            action = policy_dist_truncated.sample()
            if policy_dist_truncated.probs.size() != policy_dist.probs.size():
                action = torch.gather(valid_actions, 1, action.view(1, 1))
        elif self.sample_action == 'dist': #TODO: debug here.
            action = policy_dist.sample()
            while action not in valid_actions:
                action = policy_dist.sample()
        return action

