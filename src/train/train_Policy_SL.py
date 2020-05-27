# https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
import torch
import argparse
import os
import math
import json
import numpy as np
from models.Policy_network import PolicyLSTM, PolicyMLP
from data_provider.CLEVR_Dataset import CLEVR_Dataset
from train.train_functions import train_one_epoch_policy, evaluate_policy
from torch.utils.data import DataLoader
from utils.utils_train import create_logger, write_to_csv

'''
training script for training the policy network with supervised learning. 
'''

if __name__ == '__main__':

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

    parser = argparse.ArgumentParser()

    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-word_emb_size", type=int, default=16, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=32, help="dimension of the hidden state")
    parser.add_argument("-p_drop", type=float, default=0, help="dropout rate")
    parser.add_argument("-grad_clip", type=float)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-bs", type=int, default=8, help="batch size")
    parser.add_argument("-ep", type=int, default=5, help="number of epochs")
    parser.add_argument("-data_path", type=str, required=True)
    parser.add_argument("-out_path", type=str, required=True)
    parser.add_argument('-num_workers', type=int, default=0, help="num workers for DataLoader")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################################################
    # Build CLEVR Dataset.
    ###############################################################################

    train_questions_path = os.path.join(args.data_path, "train_questions.h5")
    train_feats_path = os.path.join(args.data_path, 'train_features.h5')
    val_questions_path = os.path.join(args.data_path, "val_questions.h5")
    val_feats_path = os.path.join(args.data_path, 'val_features.h5')
    vocab_path = os.path.join(args.data_path, "vocab.json")

    if device.type == 'cpu':
        train_dataset = CLEVR_Dataset(h5_questions_path=train_questions_path,
                                      h5_feats_path=train_feats_path,
                                      vocab_path=vocab_path,
                                      max_samples=21)
        val_dataset = CLEVR_Dataset(h5_questions_path=val_questions_path,
                                    h5_feats_path=val_feats_path,
                                    vocab_path=vocab_path,
                                    max_samples=21)
    else:
        train_dataset = CLEVR_Dataset(h5_questions_path=train_questions_path,
                                      h5_feats_path=train_feats_path,
                                      vocab_path=vocab_path)
        val_dataset = CLEVR_Dataset(h5_questions_path=val_questions_path,
                                h5_feats_path=val_feats_path,
                                vocab_path=vocab_path)

    num_tokens = train_dataset.len_vocab
    BATCH_SIZE = args.bs
    PAD_IDX = train_dataset.vocab_questions["<PAD>"]

    train_generator = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, drop_last=True,
                                 num_workers=args.num_workers)
    val_generator = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=args.num_workers)

    ###############################################################################
    # Build the model
    ###############################################################################
    policy_network = PolicyLSTM(num_tokens=num_tokens,
                                    word_emb_size=args.word_emb_size,
                                    emb_size=args.word_emb_size + args.word_emb_size * 7 * 7,
                                    hidden_size=args.hidden_size,
                                    num_layers=args.num_layers,
                                    p_drop=args.p_drop,
                                    rl=False).to(device)

    learning_rate = args.lr
    optimizer = torch.optim.Adam(params=policy_network.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss(ignore_index=PAD_IDX)
    EPOCHS = args.ep

    num_batches = int(len(train_dataset) / args.bs)
    print_interval = 10 if device.type =='cpu' else int(num_batches / 10)

    ###############################################################################
    # Create logger, output_path and config file.
    ###############################################################################

    out_path = 'SL_LSTM_L_{}_emb_{}_hid_{}_pdrop_{}_gc_{}_bs_{}_lr_{}'.format(args.num_layers,
                                                                                       args.word_emb_size, args.hidden_size,
                                                                                       args.p_drop, args.grad_clip,
                                                                                       args.bs, learning_rate)
    out_path = os.path.join(args.out_path, out_path)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    out_file_log = os.path.join(out_path, 'training_log.log')
    logger = create_logger(out_file_log)
    out_csv = os.path.join(out_path, 'train_history.csv')
    model_path = os.path.join(out_path, 'model.pt')
    config_path = os.path.join(out_path, 'config.json')

    hparams = {}
    hparams["emb_size"] = args.word_emb_size
    hparams["hidden_size"] = args.hidden_size
    hparams["p_drop"] = args.p_drop
    hparams["grad_clip"] = args.grad_clip
    hparams["BATCH_SIZE"] = BATCH_SIZE
    hparams["learning_rate"] = learning_rate
    config = {"hparams": hparams}

    with open(config_path, mode='w') as f:
        json.dump(config, f)

    ################################################################################################################################################
    # Train the model
    ################################################################################################################################################
    logger.info('LSTM with projection layer...')
    logger.info("start training...")
    logger.info("hparams: {}".format(hparams))
    train_loss_history, train_ppl_history, val_loss_history, val_ppl_history = [], [], [], []
    logger.info('checking shape of and values of a sample of the train dataset...')
    idx = list(np.random.randint(0, train_dataset.__len__(), size=1))
    (temp_inp, temp_tar), temp_feats, _ = train_dataset.__getitem__(idx)
    logger.info('samples of input questions: {}'.format(temp_inp.data.numpy()))
    logger.info('samples of target questions: {}'.format(temp_tar.data.numpy()))
    logger.info('shape of img feats: {}'.format(temp_feats.shape))
    logger.info('train dataset length: {}'.format(train_dataset.__len__()))
    logger.info('number of tokens: {}'.format(num_tokens))
    best_val_loss = None
    for epoch in range(EPOCHS):
        logger.info('epoch {}/{}'.format(epoch + 1, EPOCHS))
        train_loss, elapsed = train_one_epoch_policy(model=policy_network,
                                              train_generator=train_generator,
                                              optimizer=optimizer,
                                              criterion=criterion,
                                              device=device,
                                              args=args,
                                              print_interval=print_interval)
        logger.info('train loss {:5.3f} - train perplexity {:8.3f}'.format(train_loss, math.exp(train_loss)))
        logger.info('time for one epoch...{:5.2f}'.format(elapsed))
        val_loss = evaluate_policy(model=policy_network, val_generator=val_generator, criterion=criterion, device=device)
        logger.info('val loss: {:5.3f} - val perplexity: {:8.3f}'.format(val_loss, math.exp(val_loss)))

        # saving loss and metrics information.
        train_loss_history.append(train_loss)
        train_ppl_history.append(math.exp(train_loss))
        val_loss_history.append(val_loss)
        val_ppl_history.append(math.exp(val_loss))
        logger.info('-' * 89)

        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or val_loss < best_val_loss:
            with open(model_path, 'wb') as f:
                torch.save(policy_network, f)
            best_val_loss = val_loss

    logger.info("saving loss and metrics information...")
    hist_keys = ['train_loss', 'train_ppl', 'val_loss', 'val_ppl']
    hist_dict = dict(zip(hist_keys, [train_loss_history, train_ppl_history, val_loss_history, val_ppl_history]))
    write_to_csv(out_csv, hist_dict)
