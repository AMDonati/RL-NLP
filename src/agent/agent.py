import torch.optim as optim


class Agent:
    def __init__(self, model, env, gamma=1., lr=1e-2, pretrained_lm=None):
        self.policy = model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.pretrained_lm = pretrained_lm
        self.env = env

    def select_action(self, state, forced=None, num_truncated=10):
        pass

    def finish_episode(self):
        pass

    def learn(self, env, writer, output_path="lm", log_interval=10, num_episodes=100, pretrain=False,
              num_truncated=10):
        pass
