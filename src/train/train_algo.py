# https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
import json
import math
import os
import numpy as np
import torch
from torch.utils.data import DataLoader
from train.train_functions import *
from utils.utils_train import create_logger, write_to_csv


class SLAlgo:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
        self.model = model
        self.train_dataset, self.val_dataset, self.test_dataset = train_dataset, val_dataset, test_dataset
        self.train_generator = DataLoader(dataset=train_dataset, batch_size=args.bs, drop_last=True, num_workers=args.num_workers)
        self.val_generator = DataLoader(dataset=val_dataset, batch_size=args.bs, drop_last=True,
                                          num_workers=args.num_workers)
        self.batch_size = args.bs
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(params=model.parameters(), lr=self.lr)
        PAD_IDX = train_dataset.vocab_questions["<PAD>"]
        self.criterion = torch.nn.NLLLoss(ignore_index=PAD_IDX)
        self.EPOCHS = args.ep
        self.grad_clip = args.grad_clip
        self.print_interval = args.print_interval
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.create_out_path(args)
        self.train_function, self.eval_function = self.get_algo_functions(args)


    def create_out_path(self, args):
        out_path = '{}_layers_{}_emb_{}_hidden_{}_pdrop_{}_gradclip_{}_bs_{}_lr_{}'.format(args.model, args.num_layers,
                                                                                           args.emb_size,
                                                                                           args.hidden_size,
                                                                                           args.p_drop, args.grad_clip,
                                                                                           args.bs, args.lr)
        out_path = os.path.join(args.out_path, out_path)
        if not os.path.exists(out_path):
            os.makedirs(out_path)
        out_file_log = os.path.join(out_path, 'training_log.log')
        self.logger = create_logger(out_file_log)
        self.out_csv = os.path.join(out_path, 'train_history.csv')
        self.model_path = os.path.join(out_path, 'model.pt')
        self.logger.info("hparams: {}".format(vars(args)))
        self.logger.info('train dataset length: {}'.format(self.train_dataset.__len__()))
        self.logger.info('number of tokens: {}'.format(self.train_dataset.len_vocab))
        self._save_hparams(args, out_path)

    def _save_hparams(self, args, out_path):
        dict_hparams = vars(args)
        dict_hparams = {key: str(value) for key, value in dict_hparams.items()}
        config_path = os.path.join(out_path, "config.json")
        with open(config_path, 'w') as fp:
            json.dump(dict_hparams, fp, sort_keys=True, indent=4)

    def get_algo_functions(self, args):
        if args.task == "lm":
            train_function = train_one_epoch_vqa if args.dataset == "vqa" else train_one_epoch
            eval_function = evaluate_vqa if args.dataset == "vqa" else evaluate
        elif args.task == "policy":
            train_function = train_one_epoch_policy
            eval_function = evaluate_policy
        return train_function, eval_function

    def train(self):
        self.logger.info("start training...")
        train_loss_history, train_ppl_history, val_loss_history, val_ppl_history = [], [], [], []
        best_val_loss = None
        for epoch in range(self.EPOCHS):
            self.logger.info('epoch {}/{}'.format(epoch + 1, self.EPOCHS))
            train_loss, elapsed = self.train_function(model=self.model,
                                                 train_generator=self.train_generator,
                                                 optimizer=self.optimizer,
                                                 criterion=self.criterion,
                                                 device=self.device,
                                                 grad_clip=self.grad_clip,
                                                 print_interval=self.print_interval)
            self.logger.info('train loss {:5.3f} - train perplexity {:8.3f}'.format(train_loss, math.exp(train_loss)))
            self.logger.info('time for one epoch...{:5.2f}'.format(elapsed))
            val_loss = self.eval_function(model=self.model, val_generator=self.val_generator, criterion=self.criterion, device=self.device)
            self.logger.info('val loss: {:5.3f} - val perplexity: {:8.3f}'.format(val_loss, math.exp(val_loss)))

            # saving loss and metrics information.
            train_loss_history.append(train_loss)
            train_ppl_history.append(math.exp(train_loss))
            val_loss_history.append(val_loss)
            val_ppl_history.append(math.exp(val_loss))
            self.logger.info('-' * 89)

            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(self.model_path, 'wb') as f:
                    torch.save(self.model, f)
                best_val_loss = val_loss

        self.logger.info("saving loss and metrics information...")
        hist_keys = ['train_loss', 'train_ppl', 'val_loss', 'val_ppl']
        hist_dict = dict(zip(hist_keys, [train_loss_history, train_ppl_history, val_loss_history, val_ppl_history]))
        write_to_csv(self.out_csv, hist_dict)

# class PolicyAlgo(SLAlgo)
#     def __init__(self, model, train_dataset, val_dataset, test_dataset, args):
#         super(PolicyAlgo, self).__init__(model=model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, args=args)



