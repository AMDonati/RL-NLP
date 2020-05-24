import logging

import torch


def test(env, writer, agent, saved_path="model.pth", log_interval=1, num_episodes=10):
    trained_model = agent.model
    trained_model.load_state_dict(torch.load(saved_path))
    trained_model.eval()

    running_reward = 0
    for i_episode in range(num_episodes):
        state, ep_reward = env.reset(), 0
        top_words = []
        for t in range(0, env.max_len + 1):
            action, log_probs, value, valid_actions, dist = agent.select_action(state)
            state_decoded = env.clevr_dataset.idx2word(state.text.numpy()[0])
            top_k_weights, top_k_indices = torch.topk(dist.probs, 10, sorted=True)
            top_words_decoded = env.clevr_dataset.idx2word(top_k_indices.numpy()[0])
            # top = " ".join(
            #    ["{}/{}".format(token, weight) for token, weight in zip(top_words_decoded.split(), top_k_weights.numpy())])
            top_words.append("next 10 possible words for {} : {}".format(state_decoded, top_words_decoded))
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
            writer.add_text('episode_questions', '  \n'.join(env.ref_questions_decoded))
            writer.add_scalar('test_running_return', running_reward, i_episode + 1)
            writer.add_text('language_model', '  \n'.join(top_words))
