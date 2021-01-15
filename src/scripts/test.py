import argparse
import configparser
import os

from scripts.run import run, get_parser as get_run_parser


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-models_path", type=str, required=True,
                        help="data folder containing models")
    parser.add_argument("-num_diversity", type=int, default=None, help="repeating on test the same image/answer")
    parser.add_argument("-num_episodes_test", type=int, default=None, help="number of test episodes")
    parser.add_argument("-test_metrics", nargs='+', type=str,
                        default=["return", "dialog", "bleu", "ppl_dialog_lm", "vilbert", "size_valid_actions",
                                 "action_probs_truncated", "ttr_question", "sum_probs", "ppl", "lv_norm", "ttr",
                                 "selfbleu", "dialogimage", "valid_actions"], help="test metrics")
    return parser


def cast_value(value):
    if value == "None":
        return None
    try:
        val = float(value)
        if val.is_integer():
            val = int(val)
    except:
        try:
            val = [val0.split("'")[1] for val0 in value.split(",")]
        except:
            val = value
    finally:
        return val


def eval(args):
    dirs = [f.path for f in os.scandir(args.models_path) if f.is_dir()]
    for dir in dirs:
        if os.path.exists(os.path.join(dir, "conf.ini")) and os.path.exists(os.path.join(dir, "model.pth")):
            conf_path = os.path.join(dir, "conf.ini")
            config = configparser.ConfigParser()
            config.read(conf_path)
            # defaults = {}
            # defaults.update(dict(config.items("main")))
            defaults = {key: cast_value(value) for key, value in dict(config.items("main")).items()}
            defaults["num_episodes_train"] = 0
            defaults["K_epochs"] = defaults["k_epochs"]
            defaults["old_policy_path"] = defaults["policy_path"]
            defaults["policy_path"] = os.path.join(dir, "model.pth")
            if args.num_diversity is not None:
                defaults["num_diversity"] = args.num_diversity
            if args.num_episodes_test is not None:
                defaults["num_episodes_test"] = args.num_episodes_test
            if args.test_metrics is not None:
                defaults["test_metrics"] = args.test_metrics

            if "temp_factor" in defaults.keys():
                if "inv_schedule_step" in defaults.keys():
                    if defaults["inv_schedule_step"] == 0:
                        if defaults["temp_factor"] < 1:
                            defaults["temperature"] = defaults["temp_min"]
                        elif defaults["temp_factor"] > 1:
                            defaults["temperature"] = defaults["temp_max"]
                    else:
                        if defaults["temp_factor"] > 1:
                            defaults["temperature"] = defaults["temp_min"]
                        elif defaults["temp_factor"] < 1:
                            defaults["temperature"] = defaults["temp_max"]
                else:
                    if defaults["temp_factor"] < 1:
                        defaults["temperature"] = defaults["temp_min"]
                    elif defaults["temp_factor"] > 1:
                        defaults["temperature"] = defaults["temp_max"]

            conf_parser = get_run_parser()
            conf_parser.set_defaults(**defaults)
            # args_list=list(np.array(list(defaults.items())).ravel())
            run_args = conf_parser.parse_known_args()
            print(run_args)
            run(run_args[0])


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    eval(args)
