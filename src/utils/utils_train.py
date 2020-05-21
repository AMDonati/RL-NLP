import csv
import logging


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
