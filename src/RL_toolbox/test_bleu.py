from RL_toolbox.reward import Bleu_sf0 as Bleu
import numpy as np
import random


def simple_test(bleu_fn):
    rew_1, _, _ = bleu_fn.get(question="The cat is on the mat", ep_questions_decoded=["The cat is on the mat"], done=True)
    print("BLEU score for equal sentences", rew_1)
    rew_2, _, _ = bleu_fn.get(question="The cat is on the mat the cat is on the mat", ep_questions_decoded=["The cat is on the mat"], done=True)
    print("BLEU score for equal sentences with double repetition in target sentence", rew_2)
    rew_3, _, _ = bleu_fn.get(question="The cat is on the mat what kinds of birds are flying",
                        ep_questions_decoded=["The cat is on the mat"], done=True)
    print("BLEU score for equal sentence + unsimilar sentence in target sentence", rew_3)
    rew_4, _, _ = bleu_fn.get(question="The cat is on the mat", ep_questions_decoded=["the the the the the the"],
                           done=True)
    print("BLEU score for degenerated target sentence", rew_4)
    rew_0, _, _ = bleu_fn.get(question="The cat is on the mat", ep_questions_decoded=["What kinds of birds are flying"],
                              done=True)
    print("BLEU score for sentences with no common words", rew_0)


def create_degenerated_text(text, dataset, num_words):
    vocab = dataset.vocab_questions
    text_words = text.split()
    new_words_id = random.sample(list(range(0, len(text_words))), k=min(num_words, len(text_words)))
    degenerated_words = []
    for _ in range(min(num_words, len(text_words))):
        word_idx = random.sample(list(vocab.values()), k=1)
        degenerated_words.append(dataset.question_tokenizer.decode(word_idx))
    for i, id in enumerate(new_words_id):
        text_words[id] = degenerated_words[i]
    return " ".join(text_words), min(num_words, len(text_words))

def test_samples_from_dataset(bleu_fn, dataset, num_words=[1, 2, 3], num_samples=1):
    dict = {}
    for _ in range(num_samples):
        rand_id = np.random.randint(0, len(dataset.filtered_entries))
        question = dataset.filtered_entries[rand_id]["question"]
        for w in num_words:
            degen_question, w_ = create_degenerated_text(question, dataset, w)
            score, _, _ = bleu_fn.get(question=degen_question,
                        ep_questions_decoded=[question], done=True)
            print("question:{} - degen question: {}".format(question, degen_question))
            print("BLEU score for degenerated question with {} degenerated words".format(w_), score)
            dict.update({str(w_): score})
    return dict


if __name__ == '__main__':
    import argparse
    import torch
    import os
    from data_provider.QuestionsDataset import QuestionsDataset
    from data_provider.CLEVR_Dataset import CLEVR_Dataset
    from data_provider.vqa_dataset import VQADataset
    from data_provider.vqa_tokenizer import VQATokenizer
    from data_provider._image_features_reader import ImageFeaturesH5Reader
    from transformers import BertTokenizer, GPT2Tokenizer

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default='../../data/vqa-v2')
    parser.add_argument("-features_path", type=str, default='../../data/vqa-v2/coco_trainval.lmdb')
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    lm_tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    reward_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
    images_feature_reader = ImageFeaturesH5Reader(args.features_path, False)
    question_tokenizer = VQATokenizer(lm_tokenizer=lm_tokenizer)

    train_dataset = VQADataset(split="mintrain", dataroot=args.data_path,
                               question_tokenizer=question_tokenizer,
                               image_features_reader=images_feature_reader,
                               reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                               num_images=None, vocab_path=os.path.join(args.data_path, 'cache/vocab.json'),
                               filter_entries=True, rl=False)
    sf_ids = [0,1,2,3,4,5,7] # 0 corresponds to function smoothing.
    print("------------------------------------------------------------------------------------------------")
    print("simple test")
    for sf_id in sf_ids:
        bleu_fn = Bleu(sf_id=sf_id)
        print("<-------------------------------------------------->")
        print("BLEU scores for smoothing function: {}".format(sf_id))
        simple_test(bleu_fn)
        print("<-------------------------------------------------->")
    print("------------------------------------------------------------------------------------------------")
    print("test degerated sample from VQA dataset")
    sf_ids = [1, 2, 3, 4]
    for sf_id in sf_ids:
        print("<-------------------------smoothing function {}------------------------->".format(sf_id))
        bleu_fn = Bleu(sf_id=sf_id)
        results = test_samples_from_dataset(bleu_fn, train_dataset)
        print("<-------------------------------------------------->")