import argparse
import os
import pandas as pd
import re

column_regex = "(?P<test_mode>.*)(?P<sampling>sampling|greedy|ranking_lm)(?P<metric>.*)"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True,
                        help="data folder containing experiments")
    parser.add_argument('-columns_to_save', nargs='+', type=str,
                        default=["oracle_score", "recall_5", "bleu", "meteor", "cider", "ppl_dialog_lm","ppl_dialog_lm_ext",
                                 "language_score", "selfbleu", "kurtosis", "peakiness"], help="")
    parser.add_argument('-bottom_folder', type=int, default=1)
    parser.add_argument('-top_folder', type=int, default=1)
    parser.add_argument('-precision', type=int, default=4)

    return parser


def merge_one_experiment(args):
    dirs = [f.path for f in os.scandir(args.path) if f.is_dir()]
    df_with_trunc = pd.DataFrame()
    df_no_trunc = pd.DataFrame()
    for dir_conf in dirs:
        name_conf = os.path.basename(dir_conf)
        dirs = [f.path for f in os.scandir(dir_conf) if f.is_dir()]
        for dir_experiment in dirs:
            name_exp = os.path.basename(dir_experiment)
            all_metrics_path = os.path.join(dir_experiment, "all_metrics.csv")
            stat_path = os.path.join(dir_experiment, "stats")
            if os.path.exists(all_metrics_path) and os.path.exists(stat_path):
                root_stat, _, stats_files = next(os.walk(stat_path))
                for stat_file in stats_files:
                    if not stat_file.endswith('_div.csv'):
                        stat_name = stat_file.split(".")[0]
                        df_stat = pd.read_csv(os.path.join(root_stat, stat_file))
                        df_stat["conf"] = name_conf
                        df_stat["exp"] = name_exp
                        df_stat[["test", "sampling", "metric"]] = df_stat[df_stat.columns[0]].str.extract(column_regex,
                                                                                                          expand=True)
                        df_stat["metric"] = df_stat["metric"].str.strip("_")
                        df_stat.replace(r'^\s*$', stat_name, regex=True, inplace=True)
                        del df_stat[df_stat.columns[0]]
                        if "with_trunc" in df_stat.columns:
                            with_trunc = df_stat[[col for col in df_stat.columns if col != "no_trunc"]]
                            df_with_trunc = df_with_trunc.append(with_trunc, ignore_index=True)
                        if "no_trunc" in df_stat.columns:
                            no_trunc = df_stat[[col for col in df_stat.columns if col != "with_trunc"]]
                            df_no_trunc = df_no_trunc.append(no_trunc, ignore_index=True)
    # str_mean_std = lambda x: str(round(x.mean(), args.precision)) + "+-" + str(round(x.std(), 2))

    if not df_no_trunc.empty:
        df_no_trunc["no_trunc"] = df_no_trunc["no_trunc"]
        df_no_trunc = df_no_trunc.pivot(index=['conf', 'exp', 'test', 'sampling'], columns='metric',
                                        values='no_trunc')
        columns_to_save = [col for col in args.columns_to_save if col in df_no_trunc.columns]

        df_no_trunc = df_no_trunc[columns_to_save]

        df_no_trunc_grouped = df_no_trunc.groupby(["conf", "sampling"]).mean()

        df_no_trunc_recap = df_no_trunc_grouped.groupby(["conf"]).mean().round(args.precision).astype(
            str) + "+-" + df_no_trunc_grouped.groupby(["conf"]).std().round(args.precision).astype(str)

        df_no_trunc.round(args.precision).to_csv(os.path.join(args.path, "stats_no_trunc.csv"))
        df_no_trunc.round(args.precision).to_latex(os.path.join(args.path, "stats_no_trunc.txt"))

        df_no_trunc_grouped.round(args.precision).to_csv(os.path.join(args.path, "stats_no_trunc_grouped.csv"))
        df_no_trunc_grouped.round(args.precision).to_latex(os.path.join(args.path, "stats_no_trunc_grouped.txt"))

        df_no_trunc_recap.round(args.precision).to_csv(os.path.join(args.path, "stats_no_trunc_recap.csv"))
        df_no_trunc_recap.round(args.precision).to_latex(os.path.join(args.path, "stats_no_trunc_recap.txt"))

    if not df_with_trunc.empty:
        df_with_trunc["with_trunc"] = df_with_trunc["with_trunc"]
        df_with_trunc = df_with_trunc.pivot(index=['conf', 'exp', 'test', 'sampling'], columns='metric',
                                            values='with_trunc')
        columns_to_save = [col for col in args.columns_to_save if col in df_with_trunc.columns]
        df_with_trunc = df_with_trunc[columns_to_save]

        df_with_trunc_grouped = df_with_trunc.groupby(["conf", "sampling"]).mean()

        df_with_trunc_recap = df_with_trunc_grouped.groupby(["conf"]).mean().round(args.precision).astype(
            str) + "+-" + df_with_trunc_grouped.groupby(["conf"]).std().round(args.precision).astype(str)

        df_with_trunc.round(args.precision).to_csv(os.path.join(args.path, "stats_with_trunc.csv"))
        df_with_trunc.round(args.precision).to_latex(os.path.join(args.path, "stats_with_trunc.txt"))

        df_with_trunc_grouped.round(args.precision).to_csv(os.path.join(args.path, "stats_with_trunc_grouped.csv"))
        df_with_trunc_grouped.round(args.precision).to_latex(os.path.join(args.path, "stats_with_trunc_grouped.txt"))

        df_with_trunc_recap.round(args.precision).to_csv(os.path.join(args.path, "stats_with_trunc_recap.csv"))
        df_with_trunc_recap.round(args.precision).to_latex(os.path.join(args.path, "stats_with_trunc_recap.txt"))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.bottom_folder == 1:
        merge_one_experiment(args)
