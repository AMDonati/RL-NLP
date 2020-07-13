import datetime
import time

import numpy as np
import pandas as pd
from hyperopt import fmin, tpe, STATUS_OK, Trials, hp

from scripts.run import get_parser, run


def objective(params):
    generic_args = {}
    params = {**params, **generic_args}
    for key, value in params.items():
        setattr(args, key, value)
    print(args)
    agent = run(args)
    rewards = agent.train_metrics["reward"].metric
    loss = -np.mean(rewards[-agent.log_interval:])
    # run()
    return {
        'loss': loss,
        'status': STATUS_OK,
        'eval_time': time.time(),
        **params
    }


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()

    search_space = { # "num_truncated": hp.choice("num_truncated", [10, 20, 40, 60, 87]),
        "update_every": hp.choice("update_every", [20, 50, 100]),
        "K_epochs": hp.choice("K_epochs", [5, 10]),
    }
    trials = Trials()
    best = fmin(objective,
                space=search_space,
                algo=tpe.suggest,
                max_evals=20,
                trials=trials)

    df = pd.DataFrame.from_records(trials.results)
    df.to_csv("{}/model_selection_{}.csv".format(args.out_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S")))
    print(best)
