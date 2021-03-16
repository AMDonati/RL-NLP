# https://towardsdatascience.com/perplexity-intuition-and-derivation-105dd481c8f3
# https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
import argparse
import os
import torch

from data_provider.QuestionsDataset import QuestionsDataset
from data_provider.CLEVR_Dataset import CLEVR_Dataset
from data_provider.vqa_dataset import VQADataset
from data_provider.vqa_tokenizer import VQATokenizer
from data_provider._image_features_reader import ImageFeaturesH5Reader
from transformers import BertTokenizer, GPT2Tokenizer
from models.LM_networks import GRUModel, LSTMModel, LayerNormLSTMModel
from models.rl_basic import PolicyLSTMBatch_SL
from train.train_algo import SLAlgo
import copy
from torch.utils.data import Subset

'''
training script for LM network. 
Inspired from: https://github.com/pytorch/examples/blob/master/word_language_model/main.py
'''


def copy_attributes(source, target):
    for attr in source.__dict__.keys():
        setattr(target, attr, getattr(source, attr))
    return target


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
    parser.add_argument("-model_path", type=str, help="path for loading the model if starting from a pre-trained model")
    parser.add_argument("-min_data", type=int, default=0,
                        help="for VQAv2 train on a subpart of the dataset and a reduced vocabulary.")
    # model params.
    parser.add_argument("-model", type=str, default="lstm", help="rnn model")
    parser.add_argument("-num_layers", type=int, default=1, help="num layers for language model")
    parser.add_argument("-emb_size", type=int, default=512, help="dimension of the embedding layer")
    parser.add_argument("-hidden_size", type=int, default=512, help="dimension of the hidden state")
    parser.add_argument("-attention_dim", type=int, default=512, help="attention dim for fusion")
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
    # Evaluation:
    parser.add_argument("-eval_modes", type=str, nargs='+', default=["sampling"])
    parser.add_argument("-bleu_sf", type=int, default=2)
    # Misc.
    parser.add_argument('-range_samples', type=str, default="0,699000",
                        help="number of samples in the dataset - to train on a subset of the full dataset")
    parser.add_argument('-max_samples', type=int,
                        help="number of samples in the dataset - to train on a subset of the full dataset")
    parser.add_argument("-print_interval", type=int, default=10, help="interval logging.")
    parser.add_argument("-device_id", type=int, default=0, help="to choose the GPU for multi-GPU VM.")
    parser.add_argument("-dataset_ext", type=int, default=0, help="dataset ext")

    args = parser.parse_args()

    device = torch.device("cuda:{}".format(args.device_id) if torch.cuda.is_available() else "cpu")


    ###############################################################################
    # LOAD DATA
    ###############################################################################

    def get_datasets(args, device):
        if args.dataset == "clevr":
            if args.dataset_ext == 0:
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
                    train_dataset = CLEVR_Dataset(h5_questions_path=train_questions_path,
                                                  h5_feats_path=train_feats_path,
                                                  vocab_path=vocab_path,
                                                  max_samples=args.max_samples)
                    val_dataset = CLEVR_Dataset(h5_questions_path=val_questions_path,
                                                h5_feats_path=val_feats_path,
                                                vocab_path=vocab_path,
                                                max_samples=args.max_samples)
                    test_dataset = val_dataset

            else:
                vocab_path = os.path.join(args.data_path, "vocab.json")
                data_path = os.path.join(args.data_path, "clevr_ext")
                full_dataset = QuestionsDataset(h5_questions_path=data_path, vocab_path=vocab_path,
                                                range_samples=args.range_samples)
                train_size = int(0.9 * len(full_dataset))
                test_size = len(full_dataset) - train_size
                train_dataset, test_dataset = torch.utils.data.random_split(full_dataset, [train_size, test_size])
                train_dataset = copy_attributes(train_dataset, train_dataset.dataset)
                test_dataset = copy_attributes(test_dataset, test_dataset.dataset)
                val_dataset = copy.deepcopy(test_dataset)
        elif args.dataset == "vqa":

            lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            reward_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
            images_feature_reader = ImageFeaturesH5Reader(args.features_path, False)
            question_tokenizer = VQATokenizer(lm_tokenizer=lm_tokenizer)

            if args.min_data:
                vocab_path = os.path.join(args.data_path, 'cache/vocab_min.json')
                train_split = "mintrain"
                val_split = "mintrain" if device.type == "cpu" else "minval"
            else:
                vocab_path = os.path.join(args.data_path, 'cache/vocab.json')
                train_split = "mintrain" if device.type == "cpu" else "train"
                val_split = "mintrain" if device.type == "cpu" else "val"

            train_dataset = VQADataset(split=train_split, dataroot=args.data_path,
                                       question_tokenizer=question_tokenizer,
                                       image_features_reader=images_feature_reader,
                                       reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                                       num_images=None, vocab_path=vocab_path,
                                       filter_entries=True, rl=False)
            val_dataset = VQADataset(split=val_split, dataroot=args.data_path,
                                     question_tokenizer=question_tokenizer,
                                     image_features_reader=images_feature_reader,
                                     reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                                     num_images=None, vocab_path=vocab_path,
                                     filter_entries=True, rl=False)
            test_dataset = val_dataset

        return train_dataset, val_dataset, test_dataset


    train_dataset, val_dataset, test_dataset = get_datasets(args, device)


    ###############################################################################
    # BUILD THE MODEL
    ###############################################################################
    def get_model(args, train_dataset, device):
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
                                       num_tokens_answer=train_dataset.len_vocab_answer, device=device,
                                       attention_dim=args.attention_dim).to(
                device)
        if args.model_path is not None:
            print("Loading trained model...")
            model_ = torch.load(os.path.join(args.model_path, "model.pt"), map_location=torch.device('cpu'))
            if isinstance(model_, dict):
                model = model.load_state_dict(model_, strict=False)
                model = model.to(device)
            else:
                model = model_.to(device)
        return model


    def get_temperatures(args):
        temperatures = []
        if "sampling" in args.eval_modes:
            temperatures.append(1)
        elif "greedy" in args.eval_modes:
            temperatures.append("greedy")
        return temperatures


    ################################################################################################################################################
    # MAIN
    ################################################################################################################################################
    if args.model_path is not None:
        assert args.ep == 0, "if model path is provided, only evaluation should be done."
    model = get_model(args, train_dataset, device)
    sl_algo = SLAlgo(model=model, train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset,
                     args=args)
    if args.ep > 0:
        sl_algo.train()
    sl_algo.generate_text()
    temperatures = get_temperatures(args)
    if args.task != "lm" or args.dataset != "clevr":
        print("computing langage metrics...")
        dict_metrics = sl_algo.compute_language_metrics(temperatures=temperatures)
        sl_algo.logger.info("language metrics: {}".format(dict_metrics))
