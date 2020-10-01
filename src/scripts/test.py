import argparse
import configparser
import os

from scripts.run import run, get_parser as get_run_parser


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("-models_path", type=str, required=True,
                        help="data folder containing models")
    return parser


def cast_value(value):
    if value == "None":
        return None
    try:
        val = float(value)
        if val.is_integer():
            val = int(val)
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
