import argparse
import gym
import yaml
from easydict import EasyDict as edict
from vilbert.task_utils import LoadDatasets


class FlickrEnv(gym.Env):
    """Clevr Env"""
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(FlickrEnv, self).__init__()
        args, task_cfg = self.get_cfg()
        task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val = self.load_dataset(
            args, task_cfg)
        dataloader = list(task_dataloader_train.values())[0]
        self.iter_data_train = iter(dataloader)

    def get_cfg(self):
        parser = argparse.ArgumentParser()

        parser.add_argument(
            "--bert_model",
            default="bert-base-uncased",
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--from_pretrained",
            default="bert-base-uncased",
            type=str,
            help="Bert pre-trained model selected in the list: bert-base-uncased, "
                 "bert-large-uncased, bert-base-cased, bert-base-multilingual, bert-base-chinese.",
        )
        parser.add_argument(
            "--output_dir",
            default="save",
            type=str,
            help="The output directory where the model checkpoints will be written.",
        )
        parser.add_argument(
            "--config_file",
            default="config/bert_base_6layer_6conect.json",
            type=str,
            help="The config file which specified the model details.",
        )
        parser.add_argument(
            "--num_train_epochs",
            default=20,
            type=int,
            help="Total number of training epochs to perform.",
        )
        parser.add_argument(
            "--train_iter_multiplier",
            default=1.0,
            type=float,
            help="multiplier for the multi-task training.",
        )
        parser.add_argument(
            "--train_iter_gap",
            default=4,
            type=int,
            help="forward every n iteration is the validation score is not improving over the last 3 epoch, -1 means will stop",
        )
        parser.add_argument(
            "--warmup_proportion",
            default=0.1,
            type=float,
            help="Proportion of training to perform linear learning rate warmup for. "
                 "E.g., 0.1 = 10%% of training.",
        )
        parser.add_argument(
            "--no_cuda", action="store_true", help="Whether not to use CUDA when available"
        )
        parser.add_argument(
            "--do_lower_case",
            default=True,
            type=bool,
            help="Whether to lower case the input text. True for uncased models, False for cased models.",
        )
        parser.add_argument(
            "--local_rank",
            type=int,
            default=-1,
            help="local_rank for distributed training on gpus",
        )
        parser.add_argument(
            "--seed", type=int, default=0, help="random seed for initialization"
        )
        parser.add_argument(
            "--gradient_accumulation_steps",
            type=int,
            default=1,
            help="Number of updates steps to accumualte before performing a backward/update pass.",
        )
        parser.add_argument(
            "--fp16",
            action="store_true",
            help="Whether to use 16-bit float precision instead of 32-bit",
        )
        parser.add_argument(
            "--loss_scale",
            type=float,
            default=0,
            help="Loss scaling to improve fp16 numeric stability. Only used when fp16 set to True.\n"
                 "0 (default value): dynamic loss scaling.\n"
                 "Positive power of 2: static loss scaling value.\n",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=16,
            help="Number of workers in the dataloader.",
        )
        parser.add_argument(
            "--save_name", default="", type=str, help="save name for training."
        )
        parser.add_argument(
            "--in_memory",
            default=False,
            type=bool,
            help="whether use chunck for parallel training.",
        )
        parser.add_argument(
            "--optim", default="AdamW", type=str, help="what to use for the optimization."
        )
        parser.add_argument(
            "--tasks", default="8", type=str, help="1-2-3... training task separate by -"
        )
        parser.add_argument(
            "--freeze",
            default=-1,
            type=int,
            help="till which layer of textual stream of vilbert need to fixed.",
        )
        parser.add_argument(
            "--vision_scratch",
            action="store_true",
            help="whether pre-trained the image or not.",
        )
        parser.add_argument(
            "--evaluation_interval", default=1, type=int, help="evaluate very n epoch."
        )
        parser.add_argument(
            "--lr_scheduler",
            default="mannul",
            type=str,
            help="whether use learning rate scheduler.",
        )
        parser.add_argument(
            "--baseline", action="store_true", help="whether use single stream baseline."
        )
        parser.add_argument(
            "--resume_file", default="", type=str, help="Resume from checkpoint"
        )
        parser.add_argument(
            "--dynamic_attention",
            action="store_true",
            help="whether use dynamic attention.",
        )
        parser.add_argument(
            "--clean_train_sets",
            default=True,
            type=bool,
            help="whether clean train sets for multitask data.",
        )
        parser.add_argument(
            "--visual_target",
            default=0,
            type=int,
            help="which target to use for visual branch. \
            0: soft label, \
            1: regress the feature, \
            2: NCE loss.",
        )
        parser.add_argument(
            "--task_specific_tokens",
            action="store_true",
            help="whether to use task specific tokens for the multi-task learning.",
        )

        args = parser.parse_args()
        with open("data/flickr/vilbert_tasks.yml", "r") as f:
            task_cfg = edict(yaml.safe_load(f))
        return args, task_cfg

    def load_dataset(self, args, task_cfg):
        task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val = LoadDatasets(
            args, task_cfg, args.tasks.split("-")
        )
        return task_batch_size, task_num_iters, task_ids, task_datasets_train, task_datasets_val, task_dataloader_train, task_dataloader_val

    def reset(self):
        batch = self.iter_data_train.next()
        print(batch)


if __name__ == '__main__':
    env = FlickrEnv()
    env.reset()
