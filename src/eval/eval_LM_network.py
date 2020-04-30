import torch
import argparse
import os
from torch.utils.data import DataLoader
import json



def top_k_next_words(test_dataset, samples, k):
  return None


if __name__ == '__main__':

  parser = argparse.ArgumentParser()
  parser.add_argument("-data_path", type=str, required=True, default='../../data')

  args = parser.parse_args()

  # Load test Dataset.
  test_questions_path = os.path.join(args.data_path, "train_questions.h5")
  test_dataset = QuestionsDataset(h5_questions_path=test_questions_path, vocab_path=vocab_path)