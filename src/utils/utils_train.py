import argparse
import csv
import logging
import os
from csv import writer

import numpy as np


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def write_to_csv(output_dir, dic):
    """Write a python dic to csv."""
    with open(output_dir, 'w') as csv_file:
        writer = csv.writer(csv_file)
        for key, value in dic.items():
            writer.writerow([key, value])


def create_logger(out_file_log, level="INFO"):
    levels = {"INFO": logging.INFO, "DEBUG": logging.DEBUG, "ERROR": logging.ERROR}
    level = levels[level]
    logging.basicConfig(filename=out_file_log, level=level, filemode='w')
    # create logger
    logger = logging.getLogger('training log')
    logger.setLevel(level)
    # create console handler and set level
    ch = logging.StreamHandler()
    ch.setLevel(level)
    # create formatter
    formatter = logging.Formatter("%(asctime)s;%(levelname)s;%(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    # add formatter to ch
    ch.setFormatter(formatter)
    # add ch to logger
    logger.addHandler(ch)
    return logger


def saving_training_history(keys, values, output_path):
    history = dict(zip(keys, values))
    write_to_csv(output_path, history)


def append_list_as_row(file_name, list_of_elem):
    # Open file in append mode
    with open(file_name, 'a+', newline='') as write_obj:
        # Create a writer object from csv module
        csv_writer = writer(write_obj)
        # Add contents of list as last row in the csv file
        csv_writer.writerow(list_of_elem)


def write_to_csv_by_row(output_dir, dic):
    """Write a python dic to csv. Each Key is a column."""
    for key, value in dic.items():
        list_of_elem = [key] + value
        append_list_as_row(output_dir, list_of_elem)


def compute_write_all_metrics(agent, output_path, logger, keep=None):
    # write to csv test scalar metrics:
    all_metrics = {}
    csv_file = "all_metrics.csv" if keep is None else "all_metrics_{}.csv".format(keep)

    logger.info(
        "------------------------------------- test metrics statistics -----------------------------------------")
    for key, metric in agent.metrics.items():
        logger.info('------------------- {} -------------------'.format(key))
        # metric.write_to_csv()
        # saving the mean of all metrics in a single csv file:
        if metric.stats is not None:
            list_stats = list(metric.stats)
            all_metrics[metric.key] = np.round(np.mean(list_stats[0]), decimals=3)
    write_to_csv(os.path.join(output_path, csv_file), all_metrics)
