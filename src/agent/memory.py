import numpy as np
import torch


class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_img = []
        self.states_text = []
        self.states_answer = []
        self.logprobs = []
        self.logprobs_truncated = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.ht = []
        self.ct = []
        self.logprobs_lm = []
        self.arrs = [self.actions, self.states_text, self.states_img, self.logprobs, self.logprobs_truncated,
                     self.rewards,
                     self.is_terminals, self.values, self.states_answer, self.ht, self.ct, self.logprobs_lm]

        self.idx_episode = 0

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.states_img[:]
        del self.states_text[:]
        del self.states_answer[:]
        del self.logprobs[:]
        del self.logprobs_truncated[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]
        del self.ht[:]
        del self.ct[:]
        del self.logprobs_lm[:]

    def add_step(self, actions, states_text, states_img, logprobs, log_probs_truncated, rewards, is_terminals, values,
                 states_answer, ht, ct,  log_probas_lm):
        for arr, val in zip(self.arrs,
                            [actions, states_text, states_img, logprobs, log_probs_truncated, rewards, is_terminals,
                             values, states_answer, ht, ct, log_probas_lm]):
            arr.append(val)
