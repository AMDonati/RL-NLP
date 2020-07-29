import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical

#TODO: add logits fusion truncation function.

class Truncation:
    def __init__(self, agent, dist_action="dist_truncated"):
        self.agent = agent
        self.dist_action = dist_action

    def get_valid_actions(self, state_text, state_img):
        pass

    def get_policy_distributions(self, state, valid_actions):
        if type(self.agent).__name__ == 'PPO': #trick to distinguish between PPO and REINFORCE in select_action.
            policy_dist, policy_dist_truncated, value = self.agent.policy_old(state.text, state.img, valid_actions)
        elif type(self.agent).__name__ == 'REINFORCE':
            policy_dist, policy_dist_truncated, value = self.agent.policy(state.text, state.img, valid_actions)
        return policy_dist, policy_dist_truncated, value

    def sample_action(self, policy_dist, policy_dist_truncated, valid_actions):
        if self.dist_action == 'dist_truncated':
            action = policy_dist_truncated.sample()
            if policy_dist_truncated.probs.size() != policy_dist.probs.size():
                action = torch.gather(valid_actions, 1, action.view(1, 1))
        elif self.dist_action == 'dist':
            action = policy_dist.sample()
            while action not in valid_actions:
                action = policy_dist.sample()
        return action

class NoTruncation(Truncation):
    def __init__(self, agent, dist_action="dist_truncated", **kwargs):
        Truncation.__init__(self, agent, dist_action)
        self.dist_action = "dist_truncated"

    def get_valid_actions(self, state):
        return None, None


class TopK(Truncation):
    def __init__(self, agent, dist_action="dist_truncated", **kwargs):
        Truncation.__init__(self, agent, dist_action)
        self.num_truncated = kwargs["num_truncated"]

    def get_valid_actions(self, state):
        with torch.no_grad():
            if self.agent.lm_sl:
                seq_len = state.text.size(1)
                log_probas, _ = self.agent.pretrained_lm(state.text.to(self.agent.device))
                log_probas = log_probas.view(len(state.text), seq_len, -1)
                log_probas = log_probas[:, -1, :]
                top_k_weights, top_k_indices = torch.topk(log_probas, self.num_truncated, sorted=True)
            else:
                dist, dist_, value = self.agent.pretrained_lm(state.text, state.img)
                probs = dist.probs
                top_k_weights, top_k_indices = torch.topk(probs, self.num_truncated, sorted=True)
        return top_k_indices, top_k_weights.exp()


class ProbaThreshold(Truncation):
    def __init__(self, agent, dist_action="dist_truncated", **kwargs):
        Truncation.__init__(self, agent, dist_action)
        self.p_th = kwargs["p_th"]

    def get_valid_actions(self, state):
        with torch.no_grad():
            if self.agent.lm_sl:
                seq_len = state.text.size(1)
                log_probas, logits = self.agent.pretrained_lm(state.text.to(self.agent.device))
                logits = logits.view(len(state.text), seq_len, -1)
                probas = F.softmax(logits[:,-1,:], dim=-1)
            else:
                dist, dist_, value = self.agent.pretrained_lm(state.text, state.img)
                probas = dist.probs
            probas_mask = torch.ge(probas, self.p_th)
            valid_actions = torch.nonzero(probas_mask, as_tuple=False)[:,1] # slice trick to get only the indices.
            action_probs = probas[:,valid_actions]
            assert torch.all(action_probs >= self.p_th), "ERROR in proba threshold truncation function"
        return valid_actions.unsqueeze(0), action_probs

class SampleVA(Truncation):
    def __init__(self, agent, dist_action="dist_truncated", **kwargs):
        Truncation.__init__(self, agent, dist_action)
        self.k_min = kwargs["k_min"]
        self.k_max = kwargs["num_truncated"]

    def get_valid_actions(self, state):
        with torch.no_grad():
            if self.agent.lm_sl:
                seq_len = state.text.size(1)
                log_probas, logits = self.agent.pretrained_lm(state.text.to(self.agent.device))
                logits = logits.view(len(state.text), seq_len, -1)
                probas = F.softmax(logits[:,-1,:], dim=-1)
                dist = Categorical(probas)
            else:
                dist, _, _ = self.agent.pretrained_lm(state.text, state.img)
            valid_actions = []
            while len(valid_actions) < self.k_min:
                actions = dist.sample(sample_shape=[self.k_max])
                valid_actions = torch.unique(actions)
            action_probs = dist.probs[:,valid_actions]
        return valid_actions.unsqueeze(0), action_probs

truncations = {"no_trunc": NoTruncation, "top_k": TopK, "proba_thr":ProbaThreshold, "sample_va": SampleVA}


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
    sample_va = SampleVA(agent=agent, num_truncated=10, k_min=5, dist_action='dist')
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