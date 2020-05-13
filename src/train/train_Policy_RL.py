# https://pytorch.org/tutorials/intermediate/tensorboard_tutorial.html?highlight=tensorboard
# example of ROUGE computation: https://github.com/atulkum/pointer_summarizer/blob/master/data_util/utils.py
from models.Policy_network import PolicyLSTM, PolicyMLP
import argparse
from utils.utils_train import create_logger, write_to_csv
from train.RL_functions import *
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
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-word_emb_size", type=int, default=12, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=24, help="dimension of the hidden state")
    parser.add_argument("-p_drop", type=float, default=0, help="dropout rate")
    parser.add_argument("-grad_clip", type=float)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-bs", type=int, default=16, help="batch size")
    parser.add_argument("-max_len", type=int, default=10, help="max episode length")
    parser.add_argument("-num_training_steps", type=int, default=1000, help="number of training_steps")
    parser.add_argument("-data_path", type=str, required=True,
                        help="data folder containing questions embeddings and img features")
    parser.add_argument("-out_path", type=str, required=True, help="out folder")
    parser.add_argument('-num_workers', type=int, default=0, help="num workers for DataLoader")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################################################
    # Build CLEVR DATASET
    ###############################################################################

    h5_questions_path = os.path.join(args.data_path, 'train_questions.h5')
    h5_feats_path = os.path.join(args.data_path, 'train_features.h5')
    vocab_path = os.path.join(args.data_path, 'vocab.json')
    clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                  h5_feats_path=h5_feats_path,
                                  vocab_path=vocab_path)

    num_tokens = clevr_dataset.len_vocab
    feats_shape = clevr_dataset.feats_shape
    SOS_idx = clevr_dataset.vocab_questions["<SOS>"]
    EOS_idx = clevr_dataset.vocab_questions["<EOS>"]

    Special_Tokens = namedtuple('Special_Tokens', ('SOS_idx', 'EOS_idx'))
    special_tokens = Special_Tokens(SOS_idx, EOS_idx)
    State = namedtuple('State', ('text', 'img'))
    Episode = namedtuple('Episode', ('img_idx', 'img_feats', 'GD_questions', 'dialog', 'rewards'))

    ##################################################################################################################
    # Build the Policy Network and define hparams
    ##################################################################################################################

    policy_network = PolicyMLP(num_tokens=num_tokens,
                               word_emb_size=args.word_emb_size,
                               units=args.word_emb_size + args.word_emb_size * 7 * 7)
    # policy_network = PolicyLSTM(num_tokens=num_tokens,
    #                             word_emb_size=args.word_emb_size,
    #                             emb_size=args.word_emb_size + args.word_emb_size*7*7,
    #                             hidden_size=args.hidden_size,
    #                             num_layers=args.num_layers,
    #                             p_drop=args.p_drop)

    optimizer = torch.optim.Adam(policy_network.parameters(), lr=args.lr)

    out_file_log = os.path.join(args.out_path, 'RL_training_log.log')
    logger = create_logger(out_file_log)
    csv_out_file = os.path.join(args.out_path, 'train_history_L_1_emb_16_hid_32_drop_0_gc_None_bs_128_lr_0.001.csv')

    train_dataset = clevr_dataset
    store_episodes = True
    log_interval = 10

    #####################################################################################################################
    # REINFORCE Algo.
    #####################################################################################################################

    # -------- test of generate one episode function  ------------------------------------------------------------------------------------------------------
    # log_probs, returns, episodes = generate_one_episode(clevr_dataset=clevr_dataset,
    #                                                     policy_network=policy_network,
    #                                                     special_tokens=special_tokens,
    #                                                     device=device)
    # ------------------------------------------------------------

    running_return, sum_loss = 0., 0.
    all_episodes = []
    loss_hist, batch_return_hist, running_return_hist = [], [], []
    writer = SummaryWriter('runs/REINFORCE_CLEVR')

    for i in range(args.num_training_steps):
        log_probs_batch, returns_batch, episodes_batch = [], [], []
        for _ in range(args.bs):
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
        batch_avg_return = returns_batch[:, -1, :].mean(0).squeeze().data.numpy()
        loss = train_episodes_batch(log_probs_batch=log_probs_batch, returns_batch=returns_batch, optimizer=optimizer)
        sum_loss += loss
        running_return = 0.1 * batch_avg_return + (1 - 0.1) * running_return
        if i % log_interval == 0:
            logger.info('train loss for training step {}: {:5.3f}'.format(i, loss))
            logger.info('average batch return for training step {}: {:5.3f}'.format(i, batch_avg_return))
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

    write_to_csv(csv_out_file, hist_dict)

    # ------------------------------------------------------------
    # all_episodes, hist_dict = REINFORCE(train_dataset=clevr_dataset,
    #                          policy_network=policy_network,
    #                          special_tokens=special_tokens,
    #                          batch_size=args.bs,
    #                          optimizer=optimizer,
    #                          device=device,
    #                          num_training_steps=args.num_training_steps,
    #                          logger=logger)
