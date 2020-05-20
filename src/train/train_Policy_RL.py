# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html?highlight=tensorboard
# example of ROUGE computation: https://github.com/atulkum/pointer_summarizer/blob/master/data_util/utils.py
import os
import torch
import numpy as np
import argparse
from models.Policy_network import PolicyLSTM, PolicyMLP
from envs.clevr_env import ClevrEnv
from utils.utils_train import create_logger, write_to_csv
from RL_toolbox.RL_functions import padder_batch
from RL_toolbox.reinforce import REINFORCE
from torch.utils.tensorboard import SummaryWriter


#  trick for boolean parser args.
def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-model", type=str, default="lstm", help="lstm or mlp")
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-word_emb_size", type=int, default=32, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=64, help="dimension of the hidden state")
    parser.add_argument("-p_drop", type=float, default=0, help="dropout rate")
    parser.add_argument("-grad_clip", type=float)
    parser.add_argument("-lr", type=float, default=0.0005)
    parser.add_argument("-bs", type=int, default=1, help="batch size")
    parser.add_argument("-max_len", type=int, default=5, help="max episode length")
    parser.add_argument("-num_training_steps", type=int, default=100000, help="number of training_steps")
    parser.add_argument("-action_selection", type=str, default='sampling', help='mode to select action (greedy or sampling)')
    parser.add_argument("-data_path", type=str, required=True, help="data folder containing questions embeddings and img features")
    parser.add_argument("-out_path", type=str, required=True, help="out folder")
    parser.add_argument('-pre_train', type=str2bool, default=False, help="pre-train the policy network with SL.")
    parser.add_argument('-model_path', type=str, default='../../output/SL_32_64/model.pt', help="path for the pre-trained model with SL")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################################################
    # Load CLEVR ENVIRONMENT
    ###############################################################################

    env = ClevrEnv(data_path=args.data_path, max_len=args.max_len, max_samples=20)
    num_tokens = env.num_tokens

    ##################################################################################################################
    # Build the Policy Network and define hparams
    ##################################################################################################################
    if args.pre_train:
        print('pre-training phase...')
        assert args.model_path is not None
        with open(args.model_path, 'rb') as f:
            policy_network = torch.load(f, map_location=device).to(device)
            #TODO: add the value_head layer.
    else:
        if args.model == 'mlp':
            policy_network = PolicyMLP(num_tokens=num_tokens,
                                   word_emb_size=args.word_emb_size,
                                   units=args.word_emb_size + args.word_emb_size * 7 * 7).to(device)
        elif args.model == 'lstm':
            policy_network = PolicyLSTM(num_tokens=num_tokens,
                                    word_emb_size=args.word_emb_size,
                                    emb_size=args.word_emb_size + args.word_emb_size*7*7,
                                    hidden_size=args.hidden_size,
                                    num_layers=args.num_layers,
                                    p_drop=args.p_drop,
                                    value_fn=True).to(device)

    optimizer = torch.optim.Adam(policy_network.parameters(), lr=args.lr)
    output_path = os.path.join(args.out_path, "RL_lv_reward_{}_emb_{}_hid_{}_lr_{}_bs_{}_maxlen_{}_mode_{}".format(args.model,
                                                                                        args.word_emb_size,
                                                                                        args.hidden_size,
                                                                                        args.lr,
                                                                                        args.bs,
                                                                                        args.max_len,
                                                                                        args.action_selection))
    if not os.path.isdir(output_path):
        os.makedirs(output_path)
    out_file_log = os.path.join(output_path, 'RL_training_log.log')
    logger = create_logger(out_file_log)
    csv_out_file = os.path.join(output_path, 'train_history.csv')
    model_path = os.path.join(output_path, 'model.pt')

    log_interval = 100

    #####################################################################################################################
    # REINFORCE Algo.
    #####################################################################################################################

    reinforce = REINFORCE(env=env, model=policy_network, optimizer=optimizer, device=device, mode=args.action_selection)

    running_return, sum_loss = 0., 0.
    all_episodes = []
    loss_hist, batch_return_hist, running_return_hist = [], [], []
    writer = SummaryWriter(log_dir=os.path.join(output_path, 'runs'))

    # Get and print set of questions for the fixed img.
    logger.info('RL from scratch with word-level levenshtein reward on Image #0, with episode max length = 5 and reduce action space of 23 tokens')

    for i in range(args.num_training_steps):
        log_probs_batch, returns_batch, values_batch, episodes_batch = [], [], [], []
        for batch in range(args.bs):
            log_probs, returns, values, episode = reinforce.generate_one_episode()
            log_probs_batch.append(log_probs)
            returns_batch.append(returns)
            values_batch.append(values)
            episodes_batch.append(episode)

        # getting return statistics before padding.
        return_batch = [r[-1] for r in returns_batch]
        batch_avg_return = sum(return_batch) / len(return_batch)
        batch_max_return, max_id = max(return_batch), np.asarray(return_batch).argmax()
        max_dialog, closest_question = episodes_batch[max_id].dialog, episodes_batch[max_id].closest_question

        log_probs_batch = padder_batch(log_probs_batch) # shape (bs, max_len, 1)
        returns_batch = padder_batch(returns_batch) # shape (bs, max_len, 1)
        values_batch = padder_batch(values_batch)
        loss = reinforce.train_batch(returns=returns_batch, log_probs=log_probs_batch, values=values_batch)
        sum_loss += loss
        running_return = 0.1 * batch_avg_return + (1 - 0.1) * running_return

        if i == 0:
            logger.info('ep questions(5 tokens):')
            logger.info('\n'.join(env.ref_questions_decoded))
            writer.add_text('episode_questions', ('...').join(env.ref_questions_decoded))

        if i % log_interval == log_interval - 1:
            logger.info('train loss for training step {}: {:5.3f}'.format(i, sum_loss / log_interval))
            logger.info('average batch return for training step {}: {:5.3f}'.format(i, batch_avg_return))
            logger.info('running return for training step {}: {:8.3f}'.format(i, running_return))

            # writing metrics to tensorboard.
            writer.add_scalar('batch return', batch_avg_return, i + 1)
            writer.add_scalar('running return', running_return, i + 1)
            writer.add_scalar('training loss', sum_loss / log_interval, i+1)
            writer.add_text('best current dialog and closest question:',
                            ('------------------------').join([max_dialog, 'max batch return:' + str(batch_max_return), closest_question]),
                            global_step=i+1)

            sum_loss = 0. #resetting loss.

            with open(model_path, 'wb') as f:
                torch.save(policy_network, f)
            # save loss and return information.
            loss_hist.append(loss / log_interval)
            batch_return_hist.append(batch_avg_return)
            running_return_hist.append(running_return)
            all_episodes.append(episodes_batch)

    hist_keys = ['loss', 'return_batch', 'running_return']
    hist_dict = dict(zip(hist_keys, [loss_hist, batch_return_hist, running_return_hist]))
    write_to_csv(csv_out_file, hist_dict)

