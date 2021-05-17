import argparse
from tensorflow.python.summary.summary_iterator import summary_iterator
from pathlib import Path
from packaging import version

import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from scipy import stats
import tensorboard as tb
import os


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True,
                        help="data folder containing experiments")
    parser.add_argument("-smooth", type=int, default=1000,
                        help="data folder containing experiments")
    return parser


def plot(args):
    files = Path(args.path).rglob('*/*2021*/metrics/train_train_trunc_sampling_return*')
    rewards_all = {}
    for file in files:
        experiment = file.parent.parent.parent.name
        rewards = pd.read_csv(file, header=None)
        rewards_all[experiment] = rewards

    print("ok")
    all = pd.concat(rewards_all, axis=1)
    smoothed = all.rolling(window=args.smooth).mean()[args.smooth:]
    smoothed.reset_index(inplace=True)

    melted = smoothed.melt(id_vars=["index"], var_name="experiment")
    melted_ = melted[["index", "experiment", "value"]]
    melted_.columns=["steps", "experiment", "reward"]
    fig = plt.figure(figsize=(16, 6))
    sns.lineplot(data=melted_, x="steps", y="reward",
                 hue="experiment").set_title("Smoothed reward over training")
    fig.savefig(os.path.join(args.path, "train.png"))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    plot(args)
