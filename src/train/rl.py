import logging
import os
import random

import pandas as pd
import torch


def train(env, agent, writer, output_path="lm", log_interval=10, num_episodes=100, pretrain=False):
    # writer = SummaryWriter(log_dir=os.path.join(output_path, 'runs'))
    running_reward = 0
    for i_episode in range(num_episodes):
        state, ep_reward = env.reset(), 0
        ref_question = env.ref_questions[random.randint(0, len(env.ref_questions) - 1)]
        for t in range(0, env.max_len + 1):
            forced = ref_question[t] if pretrain else None
            action, log_probs, value = agent.select_action(state, forced=forced)
            state, (reward, _), done, _ = env.step(action)
            if pretrain:
                value = torch.tensor([-1.])
            agent.model.rewards.append(reward)
            agent.model.values.append(value)
            agent.model.saved_log_probs.append(log_probs)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        agent.finish_episode()
        if i_episode % log_interval == 0:
            logging.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            writer.add_text('episode_questions', '\n'.join(env.ref_questions_decoded))
            writer.add_scalar('train_running_return', running_reward, i_episode + 1)

        df = pd.DataFrame(agent.model.last_policy[-env.max_len:])
        # diff_df = df.diff(periods=5)
        diff_df = (df.iloc[-1] - df.iloc[0]).abs()
        top_words = diff_df.nlargest(4)
        logging.info("top words changed in the policy : {}".format(env.clevr_dataset.idx2word(top_words.index)))

    out_file = os.path.join(output_path, 'model.pth')
    # with open(out_file, 'wb') as f:
    #    torch.save(agent.model, f)
    torch.save(agent.model.state_dict(), out_file)
    return agent


def test(env, agent, writer, log_interval=1, num_episodes=10):
    running_reward = 0
    for i_episode in range(num_episodes):
        state, ep_reward = env.reset(), 0
        for t in range(0, env.max_len + 1):
            action, log_probs, value = agent.select_action(state)
            state, (reward, _), done, _ = env.step(action)
            agent.model.rewards.append(reward)
            agent.model.values.append(value)
            agent.model.saved_log_probs.append(log_probs)
            ep_reward += reward
            if done:
                break

        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward
        if i_episode % log_interval == 0:
            logging.info('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))
            writer.add_text('episode_questions', '\n'.join(env.ref_questions_decoded))
            writer.add_scalar('test_running_return', running_reward, i_episode + 1)
