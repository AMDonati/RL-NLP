import torch
import torch.optim as optim


class Agent:
    def __init__(self, model, env, gamma=1., lr=1e-2, pretrained_lm=None):
        self.policy = model
        self.optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.gamma = gamma
        self.pretrained_lm = pretrained_lm
        self.env = env

    def get_top_k_words(self, state, top_k=10):
        """
        Truncate the action space with the top k words of a pretrained language model
        :param state: state
        :param top_k: number of words
        :return: top k words
        """
        if self.pretrained_lm is None:
            return None
        dist, value = self.pretrained_lm(state.text, state.img, None)
        probs = dist.probs
        top_k_weights, top_k_indices = torch.topk(probs, top_k, sorted=True)
        valid_actions = {i: token for i, token in enumerate(top_k_indices.numpy()[0])}
        return valid_actions

    def select_action(self, state, forced=None, num_truncated=10):
        pass

    def finish_episode(self):
        pass

    def learn(self, env, writer, output_path="lm", log_interval=10, num_episodes=100, pretrain=False,
              num_truncated=10):
        pass
