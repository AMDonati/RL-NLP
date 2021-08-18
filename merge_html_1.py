import argparse
import os
import re

import pandas as pd

from gapi import get_google_api


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True,
                        help="data folder containing experiments")
    parser.add_argument('-columns_to_save', nargs='+', type=str,
                        default=["return", "oracle_score", "recall_5", "bleu", "meteor", "cider", "ppl_dialog_lm",
                                 "ppl_dialog_lm_ext", "selfbleu", "kurtosis", "peakiness",
                                 "size_valid_actions", "sum_probs_truncated"], help="")
    return parser


def merge_one_experiment(args, api):
    all_df = pd.DataFrame()
    path = os.path.join(args.path, "html")
    if not os.path.exists(path):
        os.makedirs(path)
    for root, folders, files in os.walk(args.path):
        for f in files:
            if f.endswith(".html"):
                page = open(os.path.join(root, f)).read()
                regexes = [">link : (.*?)</li", ">question : (.*?)<", ">ref_questions : (.*?)<", ">ref_answer : (.*?)<",
                           ">reward : (.*?)<", ">img : (.*?)<", ]
                data = [re.findall(reg, page) for reg in regexes]
                df = pd.DataFrame(data).transpose()
                df["root"] = os.path.basename(root)
                all_df = all_df.append(df, ignore_index=True)
                # df.set_index(["index", "root"], inplace=True)

    all_df.columns = ["link", "questions", "ref_questions", "ref_answer", "reward", "img", "root"]
    all_df_ = all_df[["link", "img", "ref_questions", "ref_answer", "root", "reward", "questions"]]

    all_df_.set_index(["img", "link", "ref_questions", "ref_answer", "root"], inplace=True)
    all_df_.sort_values(["img", "link", "ref_questions", "ref_answer", "root", "reward"], inplace=True, ascending=False)

    grouped = all_df_.groupby(["img", "link", "ref_questions", "ref_answer"])
    for name, group in grouped:
        if len(list(group.index.unique("root"))) > 3:
            group_to_save = group.reset_index()
            group_to_save["questions_"] = group_to_save.apply(
                lambda
                    x: f"\\textcolor{{Mahogany}}{{{x.questions}}}" if x.reward == "0.0" else f"\\textcolor{{PineGreen}}{{{x.questions}}}",
                axis=1)
            group_to_save = group_to_save[['root', 'questions']]
            # group_to_save.to_csv(os.path.join(path, f"{name[0]}_{name[-1]}.csv"))
            url = re.findall("<img src=(.*?)>", name[1])[0]
            name_ = name[0].zfill(6)
            img = f"img/coco/COCO_val2014_000000{name_}.jpg"
            url = f"\\href{{ici}}{{{url}}}"
            print(url)

            captions_items = {}
            captions_items["img"] = f"\\includegraphics[width=200px]{{{img}}}"
            # captions_items["url"] = f"\\href{{ici}}{{{url}}}"
            captions_items["imgid"] = name[0]
            captions_items["ref questions"] = name[2]
            captions_items["ref answer"] = name[-1]

            caption = "\\\\".join([f"{key}: {value}" for key, value in captions_items.items()])
            with pd.option_context("max_colwidth", 1000):
                group_to_save.to_latex(os.path.join(path, f"{name[0]}_{name[-1]}.txt"), label=f"{name[0]}_{name[-1]}",
                                       caption=caption, index=False)

    all_df.to_csv(os.path.join(args.path, "df.csv"))
    all_df.to_html(os.path.join(args.path, "df.html"), escape=False)


if __name__ == '__main__':
    api = get_google_api()
    parser = get_parser()
    args = parser.parse_args()
    merge_one_experiment(args, api)
