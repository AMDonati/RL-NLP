import logging

import torch
import torch.nn.functional as F
from torch.distributions import Categorical

logger = logging.getLogger()


def mask_inf_truncature(mask_valid_actions, logits, device, num_tokens=86):
    mask = (torch.ones(logits.size(0), num_tokens) * -1e32).to(device)
    mask = mask.masked_scatter_(mask_valid_actions, logits)
    probs_truncated = F.softmax(mask, dim=-1)
    policy_dist_truncated = Categorical(probs_truncated)
    return policy_dist_truncated


class Truncation:
    def __init__(self, pretrained_lm=None):
        self.language_model = pretrained_lm
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def min_max_truncation(self, probas):
        sorted_probs, sorted_indices = torch.sort(probas, descending=True)
        min_prob, max_prob = sorted_probs[:, self.s_min], sorted_probs[:, self.s_max]
        min_mask = (probas > min_prob.view(-1, 1))
        max_mask = (probas > max_prob.view(-1, 1))
        min_max_mask = torch.logical_or(min_mask, max_mask)
        return min_max_mask, sorted_probs, sorted_indices

    def get_mask(self, log_probas, logits):
        return None, None


class TopK(Truncation):
    def __init__(self, **kwargs):
        Truncation.__init__(self, pretrained_lm=kwargs["pretrained_lm"])
        self.s_min = kwargs["s_min"]
        self.s_max = kwargs["s_max"]

    def get_mask(self, log_probas, logits):
        probas = torch.exp(log_probas)
        min_max_mask, _, _ = self.min_max_truncation(probas)
        return min_max_mask


class ProbaThreshold(Truncation):
    '''See OverLeaf for details on this truncation fn.'''

    def __init__(self, **kwargs):
        Truncation.__init__(self, pretrained_lm=kwargs["pretrained_lm"])
        self.p_th = float(kwargs["truncation_params"])
        self.s_min = kwargs["s_min"]
        self.s_max = kwargs["s_max"]

    def get_mask(self, log_probas, logits):
        probas = torch.exp(log_probas)
        min_max_mask, sorted_probs, sorted_indices = self.min_max_truncation(probas)
        p_th_mask = torch.ge(probas, self.p_th)
        final_mask = torch.logical_and(min_max_mask, p_th_mask)
        return final_mask


truncations = {"top_k": TopK, "proba_thr": ProbaThreshold}
