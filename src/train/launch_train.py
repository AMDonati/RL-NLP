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
from data_provider.CLEVR_Dataset import CLEVR_Dataset
from data_provider.vqa_dataset import *
from data_provider.vqa_tokenizer import VQATokenizer
from transformers import BertTokenizer, GPT2Tokenizer
from models.LM_networks import GRUModel, LSTMModel, LayerNormLSTMModel
from models.rl_basic import PolicyLSTMBatch_SL
from train.train_functions import *
from train.train_algo import SLAlgo

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
    # task and data args.
    parser.add_argument("-task", type=str, default='lm', help="choosing between training the lm or the policy.")
    parser.add_argument("-dataset", type=str, default="vqa", help="dataset: clevr ou vqa datasets.")
    parser.add_argument("-data_path", type=str, default='../../data/vqa-v2')
    parser.add_argument("-features_path", type=str, default='../../data/vqa-v2/coco_trainval.lmdb')
    parser.add_argument("-out_path", type=str, default='../../output/temp')
    # model params.
    parser.add_argument("-model", type=str, default="lstm", help="rnn model")
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-emb_size", type=int, default=512, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=512, help="dimension of the hidden state")
    # policy network specific args.
    parser.add_argument("-kernel_size", default=1, type=int)
    parser.add_argument("-num_filters", default=3, type=int)
    parser.add_argument("-fusion", default="average", type=str)
    parser.add_argument("-stride", default=2, type=int)
    parser.add_argument('-condition_answer', type=str, default="none", help="conditioning on answer")
    # SL algo args.
    parser.add_argument("-p_drop", type=float, default=0., help="dropout rate")
    parser.add_argument("-grad_clip", type=float)
    parser.add_argument("-lr", type=float, default=0.001)
    parser.add_argument("-bs", type=int, default=2, help="batch size")
    parser.add_argument("-ep", type=int, default=3, help="number of epochs")
    parser.add_argument('-num_workers', type=int, default=0, help="num workers for DataLoader")
    # Misc.
    parser.add_argument('-range_samples', type=str, default="0,699000",
                        help="number of samples in the dataset - to train on a subset of the full dataset")
    parser.add_argument("-print_interval", type=int, default=10, help="interval logging.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    ###############################################################################
    # LOAD DATA
    ###############################################################################

    def get_datasets(args, device):
        if args.dataset == "clevr":
            train_questions_path = os.path.join(args.data_path, "train_questions.h5")
            val_questions_path = os.path.join(args.data_path, "val_questions.h5")
            test_questions_path = os.path.join(args.data_path, "test_questions.h5")
            train_feats_path = os.path.join(args.data_path, 'train_features.h5')
            val_feats_path = os.path.join(args.data_path, 'val_features.h5')
            vocab_path = os.path.join(args.data_path, "vocab.json")

            if args.task == "lm":
                train_dataset = QuestionsDataset(h5_questions_path=train_questions_path, vocab_path=vocab_path,
                                                 range_samples=args.range_samples)
                val_dataset = QuestionsDataset(h5_questions_path=val_questions_path, vocab_path=vocab_path)
                test_dataset = QuestionsDataset(h5_questions_path=test_questions_path, vocab_path=vocab_path)
            elif args.task == "policy":
                if device.type == 'cpu':
                    train_dataset = CLEVR_Dataset(h5_questions_path=train_questions_path,
                                                  h5_feats_path=train_feats_path,
                                                  vocab_path=vocab_path,
                                                  max_samples=args.max_samples)
                    val_dataset = CLEVR_Dataset(h5_questions_path=val_questions_path,
                                                h5_feats_path=val_feats_path,
                                                vocab_path=vocab_path,
                                                max_samples=args.max_samples)
                else:
                    train_dataset = CLEVR_Dataset(h5_questions_path=train_questions_path,
                                                  h5_feats_path=train_feats_path,
                                                  vocab_path=vocab_path)
                    val_dataset = CLEVR_Dataset(h5_questions_path=val_questions_path,
                                                h5_feats_path=val_feats_path,
                                                vocab_path=vocab_path)

        elif args.dataset == "vqa":
            lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            reward_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            images_feature_reader = ImageFeaturesH5Reader(args.features_path, False)
            question_tokenizer = VQATokenizer(lm_tokenizer=lm_tokenizer)

            train_split = "mintrain" if device.type == "cpu" else "train"
            val_split = "minval" if device.type == "cpu" else "val"
            train_dataset = VQADataset(split=train_split, dataroot=args.data_path,
                                       question_tokenizer=question_tokenizer,
                                       image_features_reader=images_feature_reader,
                                       reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                                       num_images=20, vocab_path=os.path.join(args.data_path, 'cache/vocab.json'),
                                       filter_entries=True, rl=False)
            val_dataset = VQADataset(split=val_split, dataroot=args.data_path,
                                     question_tokenizer=question_tokenizer,
                                     image_features_reader=images_feature_reader,
                                     reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                                     num_images=20, vocab_path=os.path.join(args.data_path, 'cache/vocab.json'),
                                     filter_entries=True, rl=False)
            test_dataset = val_dataset

        return train_dataset, val_dataset, test_dataset


    train_dataset, val_dataset, test_dataset = get_datasets(args, device)


    ###############################################################################
    # BUILD THE MODEL
    ###############################################################################
    def get_model(args, train_dataset):
        num_tokens = train_dataset.len_vocab
        if args.task == "lm":
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
        elif args.task == "policy":
            model = PolicyLSTMBatch_SL(num_tokens=num_tokens,
                                       word_emb_size=args.emb_size,
                                       hidden_size=args.hidden_size,
                                       kernel_size=args.kernel_size,
                                       num_filters=args.num_filters,
                                       stride=args.stride,
                                       fusion=args.fusion,
                                       condition_answer=args.condition_answer,
                                       num_tokens_answer=train_dataset.len_vocab_answer).to(device)
        return model

    ################################################################################################################################################
        # MAIN
    ################################################################################################################################################
    model = get_model(args, train_dataset)
    sl_algo = SLAlgo(model=model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
                     args=args)
    sl_algo.train()
    if args.task == "lm":
        sl_algo.generate_text()
