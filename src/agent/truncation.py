import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical


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
    def __init__(self, agent, sample_action="dist_truncated", p_th=0.01):
        Truncation.__init__(self, agent, sample_action)
        self.p_th = p_th

    def get_valid_actions(self, state_text, state_img, num_truncated=10):
        if self.agent.lm_sl:
            seq_len = state_text.size(1)
            log_probas, logits = self.agent.pretrained_lm(state_text.to(self.agent.device))
            logits = logits.view(len(state_text), seq_len, -1)
            probas = F.softmax(logits, dim=-1)
        else:
            dist, dist_, value = self.agent.pretrained_lm(state_text, state_img)
            probas = dist.probs
        probas_mask = probas.ge_(self.p_th)
        valid_actions, action_probs = torch.nonzero(probas_mask, as_tuple=True)
        return valid_actions, action_probs

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

class SampleVA(Truncation):
    def __init__(self, agent, sample_action="dist_truncated", k_max=20, k_min=5):
        Truncation.__init__(self, agent, sample_action)
        self.k_min = k_min
        self.k_max = k_max

    def get_valid_actions(self, state_text, state_img):
        if self.agent.lm_sl:
            seq_len = state_text.size(1)
            log_probas, logits = self.agent.pretrained_lm(state_text.to(self.agent.device))
            logits = logits.view(len(state_text), seq_len, -1)
            probas = F.softmax(logits, dim=-1)
            dist = Categorical(probas)
        else:
            dist, _, _ = self.agent.pretrained_lm(state_text, state_img)
        actions = dist.sample(sample_shape=[self.k_max])
        #TODO: if loop to check that set of actions is more than k_min.
        valid_actions = list(set(list(actions.cpu().numpy())))
        action_probs = dist.probs[valid_actions]

        return valid_actions, action_probs

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




if __name__ == '__main__':
    temp = torch.tensor([0.1, 0.2, 0.5, 0.7, 0.9])
    temp_mask = temp.ge_(0.5)
    print(temp_mask.size())

    temp_logits = torch.tensor([10, 30, -10, -40, 50], dtype=torch.float32)
    temp_dist = Categorical(logits=temp_logits)
    temp_VA = temp_dist.sample([3])
    print(temp_VA)
    import os
    from agent.ppo import PPO
    from models.rl_basic import PolicyLSTMBatch
    from envs.clevr_env import ClevrEnv
    from torch.utils.tensorboard import SummaryWriter
    data_path = "../../output/data"
    out_path = "../../output/temp"
    writer = SummaryWriter(out_path)
    env = ClevrEnv(data_path, max_len=10, reward_type="levenshtein_", mode="train", debug="0,1",
             num_questions=1, diff_reward=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    policy = PolicyLSTMBatch(env.clevr_dataset.len_vocab, 32, 64)
    pretrained_lm = torch.load(os.path.join(data_path, "best_model/model.pt"), map_location=torch.device('cpu'))
    pretrained_lm.eval()
    agent = PPO(policy=policy, env=env, test_envs=[], out_path=out_path, writer=writer)

