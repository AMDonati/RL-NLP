
class Memory:
    def __init__(self):
        self.actions = []
        self.states = []
        self.states_img = []
        self.states_text = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
        self.values = []
        self.arrs = [self.actions, self.states_text, self.states_img, self.logprobs, self.rewards,
                     self.is_terminals, self.values]

        self.idx_episode = 0

    def clear_memory(self):
        del self.actions[:]
        del self.states[:]
        del self.states_img[:]
        del self.states_text[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]
        del self.values[:]

    def add_step(self, actions, states_text, states_img, logprobs, rewards, is_terminals, values):
        for arr, val in zip(self.arrs, [actions, states_text, states_img, logprobs, rewards, is_terminals, values]):
            arr.append(val)
