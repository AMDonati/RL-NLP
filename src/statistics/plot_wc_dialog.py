from statistics.word_cloud import WordCloud
import h5py
import numpy as np
import os
from data_provider.CLEVR_Dataset import CLEVR_Dataset

out_path = "../../output/RL/2000_img_len_20/experiments/train/10-proba_thr0.05/proba_thr_0.05_eval"
dialog_path = os.path.join(out_path, "test_dialog.h5")

dialog_hf = h5py.File(dialog_path, 'r')
test_text_greedy_dialog = np.array(dialog_hf.get('test_text_greedy_with_trunc_dialog'), dtype=np.int32)

# create CLEVR dataset.
data_path = '../../data'
vocab_path = os.path.join(data_path, "vocab.json")
h5_questions_path = os.path.join(data_path, "train_questions.h5")
h5_feats_path = os.path.join(data_path, "train_features.h5")  # Caution, here train_features.h5 corresponds only to the first 21 img of the train dataset.
clevr_dataset = CLEVR_Dataset(h5_questions_path=h5_questions_path,
                                  h5_feats_path=h5_feats_path,
                                  vocab_path=vocab_path)

wc = WordCloud(path=out_path, questions=test_text_greedy_dialog, suffix='wc_test_text_greedy_dialog', dataset=clevr_dataset)