import argparse
import os
import pandas as pd


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", type=str, required=True,
                        help="data folder containing experiments")
    parser.add_argument('-columns_to_save', nargs='+', type=str,
                        default=['action_probs_truncated', 'bleu', 'lv_norm', 'ppl', 'ppl_dialog_lm',
                                 'return', 'selfbleu', 'size_valid_actions', 'sum_probs', 'ttr_question',
                                 'vilbert', 'ttr'], help="")
    return parser


def merge_one_experiment(args):
    dirs = [f.path for f in os.scandir(args.path) if f.is_dir()]

    for dir_conf in dirs:
        dirs = [f.path for f in os.scandir(dir_conf) if f.is_dir()]
        df_with_trunc = pd.DataFrame()
        df_no_trunc = pd.DataFrame()
        for dir_experiment in dirs:
            all_metrics_path = os.path.join(dir_experiment, "all_metrics.csv")
            if os.path.exists(all_metrics_path):
                df_exp = pd.read_csv(all_metrics_path, index_col=0)
                df_with_trunc = df_with_trunc.append(df_exp["with_trunc"].to_dict(), ignore_index=True)
                df_no_trunc = df_no_trunc.append(df_exp["no_trunc"].to_dict(), ignore_index=True)

        if not df_with_trunc.empty and not df_with_trunc.empty:
            str_mean_std = lambda x: str(round(x.mean(), 2)) + "+-" + str(round(x.std(), 2))
            merged_with_trunc = df_with_trunc.apply(str_mean_std)
            merged_no_trunc = df_no_trunc.apply(str_mean_std)
            all = pd.concat([merged_with_trunc, merged_no_trunc], axis=1, keys=["with_trunc", "no_trunc"])
            all = all.transpose()
            all.to_csv(os.path.join(dir_conf, "merged_metrics.csv"))
            print("ok")


def merge_all_experiments(args):
    dirs = [f.path for f in os.scandir(args.path) if f.is_dir()]
    df_with_trunc = pd.DataFrame()
    df_no_trunc = pd.DataFrame()
    for dir_conf in dirs:
        name_experiment = os.path.basename(dir_conf)
        filename = os.path.join(dir_conf, "merged_metrics.csv")
        if os.path.exists(filename):
            df = pd.read_csv(filename, index_col=0)
            df_with_trunc = df_with_trunc.append(pd.Series(df.loc["with_trunc"], name=name_experiment))
            df_no_trunc = df_no_trunc.append(pd.Series(df.loc["no_trunc"], name=name_experiment))

    columns_to_save = [col for col in args.columns_to_save if col in df_with_trunc.columns]
    df_with_trunc = df_with_trunc[columns_to_save]
    df_no_trunc = df_no_trunc[columns_to_save]

    df_with_trunc.to_csv(os.path.join(args.path, "merged_with_trunc.csv"))
    df_no_trunc.to_csv(os.path.join(args.path, "merged_no_trunc.csv"))
    df_with_trunc.to_latex(os.path.join(args.path, "merged_with_trunc.txt"))
    df_no_trunc.to_latex(os.path.join(args.path, "merged_with_trunc.txt"))


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    merge_one_experiment(args)
    merge_all_experiments(args)
