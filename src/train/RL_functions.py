import torch
import torch.nn.functional as F
import numpy as np
from collections import namedtuple
from torch.distributions import Categorical
from torch.utils.tensorboard import SummaryWriter
import os
from data_provider.CLEVR_Dataset import CLEVR_Dataset

State = namedtuple('State', ('text', 'img'))
Episode = namedtuple('Episode', ('img_idx', 'img_feats', 'GD_questions', 'dialog', 'rewards'))


def select_action(policy_network, state, device):
    policy_network.train()
    state.text.to(device)
    state.img.to(device)
    # logits, _ = policy_network(state.text, state.img) # logits > shape (s, num_tokens)
    logits = policy_network(state.text, state.img)  # logits > shape (s, num_tokens)
    probas = F.softmax(logits, dim=-1)
    m = Categorical(probas[-1, :])  # multinomial distribution with weights = probas.
    action = m.sample()
    log_prob_2 = m.log_prob(action)
    log_prob = F.log_softmax(logits, dim=-1)[-1, action]
    log_prob_3 = torch.log(probas[-1, action])
    assert abs(log_prob - log_prob_2) < 1e-5
    assert abs(log_prob - log_prob_3) < 1e-5
    return action.view(1, 1), log_prob


def get_dummy_reward(next_state_text, ep_questions, EOS_idx):
    next_state_text = next_state_text[:, 1:]  # removing sos token.
    if next_state_text[:, -1] == EOS_idx:  # remove <EOS> token if needed.
        next_state_text = next_state_text[:, :-1]
    # trunc state:
    state_len = next_state_text.size(1)
    if state_len == 0:
        print('no final reward...')
        return 0.
    else:
        # max_len = ep_questions.size(1)
        # if state_len < max_len:
        #     next_state_text = torch.cat([next_state_text, next_state_text.new_zeros(1, max_len - state_len)], dim=1)
        # assert max_len == next_state_text.size(1)
        ep_questions = ep_questions[:, :state_len]
        next_state_text = next_state_text.repeat(ep_questions.size(0), 1)
        mask_same_tokens = next_state_text == ep_questions
        reward = mask_same_tokens.sum(dim=-1).max().numpy()
        reward = reward / state_len
    return reward


# function generate one episode. debugged.
# TODO: batchify this function.
def generate_one_episode(clevr_dataset, policy_network, special_tokens, device, max_len=None, seed=None):
    if max_len is None:
        max_len = clevr_dataset.input_questions.size(
            1)  # max_length set-up to max length of questions dataset (or avg len?)
    max_len = 10  # FOR DEBUGGING.
    # sample initial state
    if seed is not None:
        np.random.seed = seed
    img_idx = np.random.randint(0, len(clevr_dataset.img_idxs))
    img_idx = 0  # FOR DEBUGGING.
    ep_GD_questions = clevr_dataset.get_questions_from_img_idx(
        img_idx)  # shape (10, S-1) # used to compute the final reward of the episode.
    img_feats = clevr_dataset.get_feats_from_img_idx(img_idx)  # shape (1024, 14, 14)
    initial_state = State(torch.LongTensor([special_tokens.SOS_idx]).view(1, 1), img_feats.unsqueeze(0))

    state = initial_state
    done = False
    step = 0
    rewards, log_probs = [], []
    while not done:
        # select the next action from the state using an epsilon greedy policy:
        action, log_prob = select_action(policy_network, state, device)
        # compute next state, done, reward from the action.
        next_state = State(torch.cat([state.text, action], dim=1), state.img)
        done = True if action.item() == special_tokens.EOS_idx or step == (max_len - 1) else False
        if done:
            reward = get_dummy_reward(next_state_text=next_state.text,
                                      ep_questions=ep_GD_questions,
                                      EOS_idx=special_tokens.EOS_idx)
        else:
            reward = 0
            step += 1
        rewards.append(reward)
        log_probs.append(log_prob)
        state = next_state

    episode = Episode(img_idx, img_feats.data.numpy(), ep_GD_questions.data.numpy(), state.text.squeeze().data.numpy(), rewards)
    return_ep = sum(rewards)
    returns = [return_ep] * (step + 1)

    if len(returns) < max_len:
        assert state.text[:, -1] == special_tokens.EOS_idx

    return log_probs, returns, episode


def padder_batch(batch):
    len_episodes = [len(l) for l in batch]
    max_len = max(len_episodes)
    batch_tensors = [torch.tensor(l, dtype=torch.float32, requires_grad=True).unsqueeze(-1) for l in
                     batch]  # tensors of shape (len_ep, 1)
    batch_tensors_padded = [torch.cat([t, t.new_zeros(max_len - len, t.size(-1))]) for (t, len) in
                            zip(batch_tensors, len_episodes)]
    batch = torch.stack(batch_tensors_padded, dim=0)
    return batch


def train_episodes_batch(log_probs_batch, returns_batch, optimizer):
    reinforce_loss = -log_probs_batch * returns_batch  # shape (bs, max_len, 1) # opposite of REINFORCE objective function to apply a gradient descent algo.
    reinforce_loss = reinforce_loss.squeeze(-1).sum(dim=1).mean(dim=0)  # sum over timesteps, mean over batches.
    optimizer.zero_grad()
    reinforce_loss.backward()
    optimizer.step()
    return reinforce_loss.item()


def REINFORCE(train_dataset, policy_network, special_tokens, args, optimizer, device, logger, log_interval=10,
              store_episodes=True):
    running_return, sum_loss = 0., 0.
    all_episodes = []
    loss_hist, batch_return_hist, running_return_hist = [], [], []
    writer = SummaryWriter('runs/REINFORCE_CLEVR')

    for i in range(args.num_training_steps):
        log_probs_batch, returns_batch, episodes_batch = [], [], []
        for _ in range(args.batch_size):
            log_probs, returns, episode = generate_one_episode(clevr_dataset=train_dataset,
                                                               policy_network=policy_network,
                                                               special_tokens=special_tokens,
                                                               max_len=args.max_len,
                                                               device=device)
            log_probs_batch.append(log_probs)
            returns_batch.append(returns)
            if store_episodes:
                episodes_batch.append(episode)

        log_probs_batch = padder_batch(log_probs_batch)
        returns_batch = padder_batch(returns_batch)
        batch_avg_return = returns_batch[:, -1, :].mean(
            0).squeeze().data.numpy()  # TODO: save the running reward and avg_return_batch and plot it.
        loss = train_episodes_batch(log_probs_batch=log_probs_batch, returns_batch=returns_batch, optimizer=optimizer)
        sum_loss += loss
        running_return = 0.1 * batch_avg_return + (1 - 0.1) * running_return
        if i % log_interval == 0:
            logger.info('train loss for training step {}: {:5.3f}'.format(i, loss))
            logger.info('running return for training step {}: {:8.3f}'.format(i, loss))
            # writing to tensorboard.
            writer.add_scalar('training loss',
                              sum_loss / (i + 1),
                              i)
            writer.add_scalar('batch return',
                              batch_avg_return,
                              i)
            writer.add_scalar('running return',
                              running_return,
                              i)
            # TODO: add the decoded batch of questions generated by the agent.
        if store_episodes:
            all_episodes.append(episodes_batch)
        # save loss and return information.
        loss_hist.append(loss)
        batch_return_hist.append(batch_avg_return)
        running_return_hist.append(running_return)

    hist_keys = ['loss', 'return_batch', 'running_return']
    hist_dict = dict(zip(hist_keys, [loss_hist, batch_return_hist, running_return_hist]))

    return all_episodes, hist_dict


if __name__ == '__main__':
    # test of get_dummy_reward function.
    h5_questions_path = os.path.join("../../data", 'train_questions.h5')
    h5_feats_path = os.path.join("../../data", 'train_features.h5')
    vocab_path = os.path.join("../../data", 'vocab.json')
    clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                  h5_feats_path=h5_feats_path,
                                  vocab_path=vocab_path)
    sample_questions = clevr_dataset.get_questions_from_img_idx(0)
    temp_state_text = torch.LongTensor([1, 7, 86, 70, 88, 21, 54, 81, 51, 84, 87, 50, 38, 17, 2]).unsqueeze(0)
    temp_reward = get_dummy_reward(temp_state_text, sample_questions, 2)
    print('reward', temp_reward)

    # ---------------------------------------- code draft ----------------------------------------------------------------------------------------
    # def get_reward(next_state_text, ep_questions, EOS_idx):
    #   # remove <EOS> token if needed.
    #   # if next_state_text[-1] == EOS_idx:
    #   #   next_state_text = next_state_text[:-1]
    #   # dialog = next_state_text.data.numpy()
    #   # ep_questions = ep_questions.data.numpy()
    #   # bools = []
    #   # for i in range(ep_questions.size(1)):
    #   #   question = ep_questions[:, i]
    #   #   if len(question) == len(dialog):
    #   #     bool = np.array_equal(question, dialog)
    #   #   else:
    #   #     bool = False
    #   #   bools.append(bool)  # TODO np.any()?
    #   return 0.
