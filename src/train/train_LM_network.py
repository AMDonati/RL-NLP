# https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
import argparse
import json
import math
import os

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_provider.QuestionsDataset import QuestionsDataset
from models.LM_networks import GRUModel, LSTMModel, LayerNormLSTMModel
from train.train_functions import train_one_epoch, evaluate
from utils.utils_train import create_logger, write_to_csv

'''
training script for LM network. 
Inspired from: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
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
    parser.add_argument("-dataset", type=str, default="clevr", help="dataset: clevr ou vqa datasets.")
    parser.add_argument("-model", type=str, default="gru", help="rnn model")
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-emb_size", type=int, required=True, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, required=True, help="dimension of the hidden state")
    parser.add_argument("-p_drop", type=float, required=True, help="dropout rate")
    parser.add_argument("-grad_clip", type=float)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-bs", type=int, default=512, help="batch size")
    parser.add_argument("-ep", type=int, default=30, help="number of epochs")
    parser.add_argument("-data_path", type=str, required=True, default='../../data')
    parser.add_argument("-out_path", type=str, required=True, default='../../output')
    parser.add_argument('-num_workers', type=int, default=0, help="num workers for DataLoader")
    parser.add_argument('-range_samples', type=str, default="0,699000",
                        help="number of samples in the dataset - to train on a subset of the full dataset")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ###############################################################################
    # Load data
    ###############################################################################

    train_questions_path = os.path.join(args.data_path, "train_questions.h5")
    val_questions_path = os.path.join(args.data_path, "val_questions.h5")
    test_questions_path = os.path.join(args.data_path, "test_questions.h5")
    vocab_path = os.path.join(args.data_path, "vocab.json")

    train_dataset = QuestionsDataset(h5_questions_path=train_questions_path, vocab_path=vocab_path,
                                     range_samples=args.range_samples)
    val_dataset = QuestionsDataset(h5_questions_path=val_questions_path, vocab_path=vocab_path)
    test_dataset = QuestionsDataset(h5_questions_path=test_questions_path, vocab_path=vocab_path)

    num_tokens = train_dataset.vocab_len
    BATCH_SIZE = args.bs
    PAD_IDX = train_dataset.get_vocab()["<PAD>"]

    train_generator = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, drop_last=True,
                                 num_workers=args.num_workers)
    val_generator = DataLoader(dataset=val_dataset, batch_size=BATCH_SIZE, drop_last=True, num_workers=args.num_workers)

    ###############################################################################
    # Build the model
    ###############################################################################
    if args.model == "gru":
        model = GRUModel(num_tokens=num_tokens,
                         emb_size=args.emb_size,
                         hidden_size=args.hidden_size,
                         num_layers=args.num_layers,
                         p_drop=args.p_drop).to(device)
    elif args.model == "lstm":
        model = LSTMModel(num_tokens=num_tokens,
                          emb_size=args.emb_size,
                          hidden_size=args.hidden_size,
                          num_layers=args.num_layers,
                          p_drop=args.p_drop).to(device)
    elif args.model == "ln_lstm":
        model = LayerNormLSTMModel(num_tokens=num_tokens,
                                   emb_size=args.emb_size,
                                   hidden_size=args.hidden_size,
                                   num_layers=args.num_layers,
                                   p_drop=args.p_drop).to(device)

    learning_rate = args.lr
    optimizer = torch.optim.Adam(params=model.parameters(), lr=learning_rate)
    criterion = torch.nn.NLLLoss(ignore_index=PAD_IDX)
    EPOCHS = args.ep

    num_batches = int(len(train_dataset) / args.bs)
    print_interval = int(num_batches / 10)

    ###############################################################################
    # Create logger, output_path and config file.
    ###############################################################################

    out_path = '{}_layers_{}_emb_{}_hidden_{}_pdrop_{}_gradclip_{}_bs_{}_lr_{}'.format(args.model, args.num_layers,
                                                                                       args.emb_size, args.hidden_size,
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
    hparams["model"] = args.model
    hparams["emb_size"] = args.emb_size
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

    logger.info("start training...")
    logger.info("hparams: {}".format(hparams))
    train_loss_history, train_ppl_history, val_loss_history, val_ppl_history = [], [], [], []
    logger.info('checking shape of and values of a sample of the train dataset...')
    idxs = list(np.random.randint(0, train_dataset.__len__(), size=2))
    temp_inp, temp_tar = train_dataset.__getitem__(idxs)
    logger.info('samples of input questions: {}'.format(temp_inp.data.numpy()))
    logger.info('samples of target questions: {}'.format(temp_tar.data.numpy()))
    logger.info('train dataset length: {}'.format(train_dataset.__len__()))
    logger.info('number of tokens: {}'.format(num_tokens))
    best_val_loss = None
    for epoch in range(EPOCHS):
        logger.info('epoch {}/{}'.format(epoch + 1, EPOCHS))
        train_loss, elapsed = train_one_epoch(model=model,
                                              train_generator=train_generator,
                                              optimizer=optimizer,
                                              criterion=criterion,
                                              device=device,
                                              args=args,
                                              print_interval=print_interval)
        logger.info('train loss {:5.3f} - train perplexity {:8.3f}'.format(train_loss, math.exp(train_loss)))
        logger.info('time for one epoch...{:5.2f}'.format(elapsed))
        val_loss = evaluate(model=model, val_generator=val_generator, criterion=criterion, device=device)
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
                torch.save(model, f)
            best_val_loss = val_loss

    logger.info("saving loss and metrics information...")
    hist_keys = ['train_loss', 'train_ppl', 'val_loss', 'val_ppl']
    hist_dict = dict(zip(hist_keys, [train_loss_history, train_ppl_history, val_loss_history, val_ppl_history]))
    write_to_csv(out_csv, hist_dict)
