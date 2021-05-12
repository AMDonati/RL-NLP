import argparse
import os
import pandas as pd
import configparser
import re
import numpy as np

column_regex = "(?P<test_mode>.*)(?P<sampling>sampling|greedy|ranking_lm_corrected|ranking_lm)(?P<metric>.*)"
ranking_regex = "(?P<test_mode>.*)(?P<trunc>no_trunc|with_trunc)_(?P<sampling>sampling_ranking_lm)_(?P<metric>ppl_dialog_lm_ext|language_score).csv"
metric_ranking_regex = "test_(?P<test_mode>.*)_(?P<trunc>no_trunc|with_trunc)_(?P<sampling>sampling_ranking_lm)_(?P<metric>.*).csv"


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True,
                        help="data folder containing experiments")
    parser.add_argument('-columns_to_save', nargs='+', type=str,
                        default=["oracle_score", "recall_5", "bleu", "meteor", "cider", "ppl_dialog_lm",
                                 "ppl_dialog_lm_ext"
                            , "language_score", "selfbleu", "kurtosis", "peakiness"], help="")
    parser.add_argument('-bottom_folder', type=int, default=1)
    parser.add_argument('-top_folder', type=int, default=1)
    parser.add_argument('-precision', type=int, default=4)

    return parser


def mean_stats_ranking(dir_experiment):
    config = configparser.ConfigParser()
    config.read(os.path.join(dir_experiment, "conf.ini"))
    num_episodes_test = int(config.get("main", "num_episodes_test"))
    _, _, ff = next(os.walk(os.path.join(dir_experiment, "metrics")))
    r = re.compile(ranking_regex)
    rankings = list(filter(r.findall, ff))
    indices = {}
    for f in rankings:
        re_keys = re.findall(ranking_regex, f)
        (_, trunc_mode, _, _) = re_keys[0]
        ppls = pd.read_csv(os.path.join(dir_experiment, "metrics", f), header=None).values
        if ppls.size == 10 * num_episodes_test:
            ppls = ppls.reshape(num_episodes_test, 10)
            min_ppl_indices = np.argmin(ppls, 1)
            flatten_indices = [i * 10 + ind for i, ind in enumerate(min_ppl_indices)]
            indices[trunc_mode] = flatten_indices

    r_all = re.compile(metric_ranking_regex)
    dfs = {"with_trunc": pd.DataFrame(), "no_trunc": pd.DataFrame()}
    for filename in list(filter(r_all.findall, ff)):
        try:
            re_keys_metric = re.search(metric_ranking_regex, filename)
            if re_keys_metric["metric"] not in ["dialog_image", "dialog"]:
                serie = pd.read_csv(
                    os.path.join(dir_experiment, "metrics", filename), header=None).iloc[:, 0]
                dfs[re_keys_metric["trunc"]][re_keys_metric["metric"]] = serie
                if re_keys_metric["metric"] == "oracle_recall_ranks":
                    dfs[re_keys_metric["trunc"]]["recall_5"] = (serie < 5).astype(int)
                    dfs[re_keys_metric["trunc"]]["oracle_score"] = (serie == 0).astype(int)


        except pd.errors.EmptyDataError as e:
            print(f"{filename} {e}")
    stats = pd.DataFrame()
    for trunc, indices_trunc in indices.items():
        reduced_dfs = dfs[trunc].iloc[indices_trunc]
        stats[trunc] = reduced_dfs.mean()
        # median for ppls
        # for col in reduced_dfs.columns:
        #    if "ppl" in col:
        #        stats[trunc].loc[col] = reduced_dfs[col].median()
    return indices, stats, dfs


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
                ranking_indices, ranking_stats, ranking_dfs = mean_stats_ranking(dir_experiment)
                ranking_stats_to_dfs = ranking_stats.reset_index()
                ranking_stats_to_dfs.rename(columns={"index": "metric"}, inplace=True)
                ranking_stats_to_dfs["conf"] = name_conf
                ranking_stats_to_dfs["exp"] = name_exp
                ranking_stats_to_dfs["test"] = "test_images"
                ranking_stats_to_dfs["sampling"] = "ranking_lm_corrected"
                ranking_stats_with = ranking_stats_to_dfs[
                    [col for col in ranking_stats_to_dfs.columns if col != "no_trunc"]]
                ranking_stats_no = ranking_stats_to_dfs[
                    [col for col in ranking_stats_to_dfs.columns if col != "with_trunc"]]
                df_with_trunc = df_with_trunc.append(ranking_stats_with, ignore_index=True)
                df_no_trunc = df_no_trunc.append(ranking_stats_no, ignore_index=True)

                _, _, metric_files = next(os.walk(os.path.join(dir_experiment, "metrics")))
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
        df_no_trunc["no_trunc"] = df_no_trunc["no_trunc"].round(args.precision)
        df_no_trunc = df_no_trunc.pivot(index=['conf', 'exp', 'test', 'sampling'], columns='metric',
                                        values='no_trunc')
        columns_to_save = [col for col in args.columns_to_save if col in df_no_trunc.columns]

        df_no_trunc = df_no_trunc[columns_to_save]

        df_no_trunc_grouped = df_no_trunc.groupby(["conf", "sampling"]).mean().round(args.precision)
        df_no_trunc_recap_ = df_no_trunc_grouped.iloc[
            df_no_trunc_grouped.index.get_level_values('sampling') != "ranking_lm_corrected"]
        df_no_trunc_recap = df_no_trunc_recap_.groupby(["conf"]).mean().round(args.precision).astype(
            str) + "+-" + df_no_trunc_recap_.groupby(["conf"]).std().round(args.precision).astype(str)

        df_no_trunc.to_csv(os.path.join(args.path, "stats_no_trunc.csv"))
        df_no_trunc.to_latex(os.path.join(args.path, "stats_no_trunc.txt"))

        df_no_trunc_grouped.to_csv(os.path.join(args.path, "stats_no_trunc_grouped.csv"))
        df_no_trunc_grouped.to_latex(os.path.join(args.path, "stats_no_trunc_grouped.txt"))

        df_no_trunc_recap.to_csv(os.path.join(args.path, "stats_no_trunc_recap.csv"))
        df_no_trunc_recap.to_latex(os.path.join(args.path, "stats_no_trunc_recap.txt"))

    if not df_with_trunc.empty:
        df_with_trunc["with_trunc"] = df_with_trunc["with_trunc"].round(args.precision)
        df_with_trunc = df_with_trunc.pivot(index=['conf', 'exp', 'test', 'sampling'], columns='metric',
                                            values='with_trunc')
        columns_to_save = [col for col in args.columns_to_save if col in df_with_trunc.columns]
        df_with_trunc = df_with_trunc[columns_to_save]

        df_with_trunc_grouped = df_with_trunc.groupby(["conf", "sampling"]).mean().round(args.precision).astype(
            str) + "+-" + df_with_trunc.groupby(["conf", "sampling"]).std().round(args.precision).astype(str)
        df_with_trunc_recap_ = df_with_trunc.iloc[
            df_with_trunc.index.get_level_values('sampling') != "ranking_lm_corrected"]
        df_with_trunc_recap = df_with_trunc_recap_.groupby(["conf"]).mean().round(args.precision).astype(
            str) + "+-" + df_with_trunc_recap_.groupby(["conf"]).std().round(args.precision).astype(str)

        df_with_trunc.to_csv(os.path.join(args.path, "stats_with_trunc.csv"))
        df_with_trunc.to_latex(os.path.join(args.path, "stats_with_trunc.txt"))

        df_with_trunc_grouped.to_csv(os.path.join(args.path, "stats_with_trunc_grouped.csv"))
        df_with_trunc_grouped.to_latex(os.path.join(args.path, "stats_with_trunc_grouped.txt"))

        df_with_trunc_recap.to_csv(os.path.join(args.path, "stats_with_trunc_recap.csv"))
        df_with_trunc_recap.to_latex(os.path.join(args.path, "stats_with_trunc_recap.txt"))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    if args.bottom_folder == 1:
        merge_one_experiment(args)
