import logging
import os
from logging.handlers import RotatingFileHandler

from src.data_provider.QuestionsDataset import QuestionsDataset
from src.statistics.word_cloud import WordCloud

prototypes = [WordCloud]


def create_logger(save_path, name):
    logger = logging.getLogger()
    # Debug = write everything
    logger.setLevel(logging.DEBUG)

    formatter = logging.Formatter('%(asctime)s :: %(levelname)s :: %(message)s')
    file_handler = RotatingFileHandler(save_path + '/' + name + '.stats.log', 'a', 1000000, 1)
    file_handler.setLevel(logging.DEBUG)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    steam_handler = logging.StreamHandler()
    steam_handler.setLevel(logging.DEBUG)
    logger.addHandler(steam_handler)

    return logger


def save_plots(out_dir, name):
    data_path = 'data'
    vocab_path = os.path.join(data_path, "vocab.json")
    train_questions_path = os.path.join(data_path, "train_questions.h5")

    train_dataset = QuestionsDataset(train_questions_path, vocab_path)
    stat_logger = create_logger(out_dir, name)
    for prototype in prototypes:
        p = prototype(out_dir, train_dataset, stat_logger, name)
        p.save_as_pdf()

if __name__ == '__main__':
    out_dir=os.path.join("output")
    save_plots(out_dir,"test")
