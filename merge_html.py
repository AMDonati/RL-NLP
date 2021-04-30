import argparse
import os
import pandas as pd
from bs4 import BeautifulSoup
import re


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True,
                        help="data folder containing experiments")
    parser.add_argument('-columns_to_save', nargs='+', type=str,
                        default=["return", "oracle_score", "recall_5", "bleu", "meteor", "cider", "ppl_dialog_lm",
                                 "ppl_dialog_lm_ext", "selfbleu", "kurtosis", "peakiness",
                                 "size_valid_actions", "sum_probs_truncated"], help="")
    return parser


def merge_one_experiment(args):
    all_df = pd.DataFrame()
    for root, folders, files in os.walk(args.path):
        for f in files:
            if f.endswith(".html"):
                page = open(os.path.join(root, f)).read()
                regexes = [">question : (.*?)<", ">ref_questions : (.*?)<", ">ref_answer : (.*?)<",
                           ">pred_answer : (.*?)<", ">link : (.*?)</li", ">reward : (.*?)<"]
                data = [re.findall(reg, page) for reg in regexes]
                df = pd.DataFrame(data).transpose()
                df["root"] = os.path.basename(root)
                all_df = all_df.append(df, ignore_index=True)
                # df.set_index(["index", "root"], inplace=True)
    all_df.columns = ["questions", "ref_questions", "ref_answer", "pred_answer", "link", "reward", "root"]
    all_df.set_index(["link", "ref_questions", "ref_answer", "root", ], inplace=True)
    all_df.sort_values(["link", "root", "reward"], inplace=True, ascending=False)
    all_df.to_csv(os.path.join(args.path, "df.csv"))
    all_df.to_html(os.path.join(args.path, "df.html"), escape=False)


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    merge_one_experiment(args)
