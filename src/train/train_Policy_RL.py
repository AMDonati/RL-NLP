# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html?highlight=tensorboard
# example of ROUGE computation: https://github.com/atulkum/pointer_summarizer/blob/master/data_util/utils.py
import os
import torch
import numpy as np
import torch.nn as nn
import argparse
from models.Policy_network import PolicyLSTM, PolicyMLP
from models.rl_basic import PolicyGRUWord
from envs.clevr_env import ClevrEnv
from utils.utils_train import create_logger, write_to_csv
from RL_toolbox.RL_functions import padder_batch
from RL_toolbox.reinforce import REINFORCE
from torch.utils.tensorboard import SummaryWriter
import pandas as pd

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
    parser.add_argument("-word_emb_size", type=int, default=32, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=64, help="dimension of the hidden state")
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-bs", type=int, default=1, help="batch size")
    parser.add_argument("-max_len", type=int, default=15, help="max episode length")
    parser.add_argument("-max_samples", type=int, default=21, help="max number of images for training")
    parser.add_argument("-num_training_steps", type=int, default=100000, help="number of training_steps")
    parser.add_argument("-action_selection", type=str, default='sampling', help='mode to select action (greedy or sampling)')
    parser.add_argument("-data_path", type=str, required=True, help="data folder containing questions embeddings and img features")
    parser.add_argument("-out_path", type=str, required=True, help="out folder")
    parser.add_argument('-pre_train', type=str2bool, default=False, help="pre-train the policy network with SL.")
    parser.add_argument("-trunc", type=str2bool, default=False, help="action space pruning")
    parser.add_argument('-model_path', type=str, default='../../output/SL_32_64_2/model.pt', help="path for the pre-trained model with SL")
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################################################
    # Load CLEVR ENVIRONMENT
    ###############################################################################

    env = ClevrEnv(data_path=args.data_path, max_len=args.max_len, max_samples=21)
    num_tokens = env.clevr_dataset.len_vocab

    ##################################################################################################################
    # Build the Policy Network and define hparams
    ##################################################################################################################
    if args.pre_train:
        print('pre-training phase...')
        assert args.model_path is not None
        with open(args.model_path, 'rb') as f:
            policy_network = torch.load(f, map_location=device).to(device)
        policy_network.rl = True
        policy_network.value_head = nn.Linear(policy_network.hidden_size, 1)
    else:
        policy_network = PolicyLSTM(num_tokens=num_tokens,
                                word_emb_size=args.word_emb_size,
                                emb_size=args.word_emb_size + args.word_emb_size * 7 * 7,
                                hidden_size=args.hidden_size, rl=True).to(device)
    if args.trunc:
        pretrained_lm_path = "../../output/best_model/model.pt"
    else:
        pretrained_lm_path = None


    optimizer = torch.optim.Adam(policy_network.parameters(), lr=args.lr)
    output_path = os.path.join(args.out_path, "RL_lv_reward_emb_{}_hid_{}_lr_{}_bs_{}_maxlen_{}_mode_{}".format(args.word_emb_size,
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

    log_interval = 10

    #####################################################################################################################
    # REINFORCE Algo.
    #####################################################################################################################

    reinforce = REINFORCE(env=env,
                          model=policy_network,
                          optimizer=optimizer,
                          device=device,
                          mode=args.action_selection,
                          pretrained_lm_path=pretrained_lm_path)

    running_return, sum_loss = 0., 0.
    all_episodes = []
    loss_hist, batch_return_hist, running_return_hist = [], [], []
    writer = SummaryWriter(log_dir=os.path.join(output_path, 'runs'))

    # Get and print set of questions for the fixed img.
    logger.info('pre_train: {} - trunc: {} - max episode length: {} - number of images: {}'.format(args.pre_train,
                                                                                       args.max_len,
                                                                                        args.trunc,
                                                                                       args.max_samples))

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
        if max_dialog is None:
            max_dialog = ""

        loss = reinforce.finish_episode()
        sum_loss += loss
        running_return = 0.1 * batch_avg_return + (1 - 0.1) * running_return

        # monitoring of change in most probable words.
        df = pd.DataFrame(reinforce.model.last_policy[-3:])
        diff_df = (df.iloc[-1] - df.iloc[0]).abs()
        top_words = diff_df.nlargest(4)

        if i == 0:
            #logger.info('ep questions(5 tokens):')
            #logger.info('\n'.join(env.ref_questions_decoded))
            writer.add_text('episode_questions', ('...').join(env.ref_questions_decoded))

        if i % log_interval == log_interval - 1:
            #logger.info('train loss for training step {}: {:5.3f}'.format(i, sum_loss / log_interval))
            #logger.info('average batch return for training step {}: {:5.3f}'.format(i, batch_avg_return))
            #logger.info('running return for training step {}: {:8.3f}'.format(i, running_return))
            logger.info("top words changed in the policy : {}".format(env.clevr_dataset.idx2word(top_words.index)))
            #logger.info('best current dialog and closest question:')
            logger.info('dialog:{} \n - closest question: {} \n- reward:{:5.2f}'.format(max_dialog, closest_question, batch_avg_return))
            #logger.info('current dialog:')
            #logger.info(max_dialog)
            #logger.info('closest question:')
            #logger.info(closest_question)
            logger.info('---------------------------------------------------------------------------------------->')
            #TODO: add the decode_episode function for action_space pruning.
            # writing metrics to tensorboard.
            writer.add_scalar('batch return', batch_avg_return, i + 1)
            writer.add_scalar('running return', running_return, i + 1)
            writer.add_scalar('training loss', sum_loss / log_interval, i+1)
            writer.add_text('best current dialog and closest question:',
                            ('--------').join([max_dialog, 'max batch return:' + str(batch_max_return), closest_question]),
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

