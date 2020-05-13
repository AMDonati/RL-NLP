import json
import h5py
import numpy as np
import os
import argparse

from preprocessing.text_functions import tokenize, encode, build_vocab

"""
Preprocessing script for CLEVR question files.
"""


def preprocess_questions(min_token_count, punct_to_keep, punct_to_remove, add_start_token, add_end_token,
                         json_data_path, vocab_out_path, h5_out_path):
    print('Loading Data...')
    with open(json_data_path, 'r') as f:
        questions = json.load(f)['questions']

    print('number of questions in json file: {}'.format(len(questions)))

    if os.path.isfile(vocab_out_path):
        print('Loading vocab...')
        with open(vocab_out_path, 'r') as f:
            vocab = json.load(f)
    else:
        print('Building vocab...')
        # question_token_to_idx
        list_questions = [q['question'] for q in questions]
        question_token_to_idx, start_words = build_vocab(sequences=list_questions,
                                            min_token_count=min_token_count,
                                            punct_to_keep=punct_to_keep,
                                            punct_to_remove=punct_to_remove)
        print('number of words in vocab: {}'.format(len(question_token_to_idx)))
        print('set of starting words...: {}'.format(start_words))
        # answer_token_to_idx
        if 'answer' in questions[0]:
            list_answers = [q['answer'] for q in questions]
            answer_token_to_idx, _ = build_vocab(list_answers, min_token_count=1,
                                              punct_to_keep=None, punct_to_remove=None)
        vocab = {'question_token_to_idx': question_token_to_idx,
                 'answer_token_to_idx': answer_token_to_idx}

        with open(vocab_out_path, 'w') as f:
            json.dump(vocab, f)

    print('Encoding questions...')
    input_questions_encoded, target_questions_encoded = [], []
    orig_idxs, img_idxs = [], []
    question_families, answers = [], []
    for orig_idx, q in enumerate(questions):
        # questions
        question = q['question']
        question_tokens, _ = tokenize(s=question,
                                   punct_to_keep=punct_to_keep,
                                   punct_to_remove=punct_to_remove,
                                   add_start_token=add_start_token,
                                   add_end_token=add_end_token)
        question_encoded = encode(seq_tokens=question_tokens,
                                  token_to_idx=vocab['question_token_to_idx'],
                                  allow_unk=True)
        input_question, target_question = question_encoded[:-1], question_encoded[1:]
        assert len(input_question) == len(target_question)
        input_questions_encoded.append(input_question)
        target_questions_encoded.append(target_question)
        # other objects:
        orig_idxs.append(orig_idx)
        img_idxs.append(q['image_index'])
        if 'question_family_index' in q:
            question_families.append(q['question_family_index'])
        if 'answer' in q:
            answers.append(vocab['answer_token_to_idx'][q['answer']])

    # Pad encoded questions
    max_question_length = max(len(x) for x in input_questions_encoded)
    for iqe, tqe in zip(input_questions_encoded, target_questions_encoded):
        while len(iqe) < max_question_length:
            iqe.append(vocab['question_token_to_idx']['<PAD>'])
            tqe.append(vocab['question_token_to_idx']['<PAD>'])
        assert len(iqe) == len(tqe)

    # Create h5 file
    print('Writing output...')
    input_questions_encoded = np.asarray(input_questions_encoded, dtype=np.int32)
    target_questions_encoded = np.asarray(target_questions_encoded, dtype=np.int32)
    print("input questions shape", input_questions_encoded.shape)
    print('target questions shape', target_questions_encoded.shape)
    with h5py.File(h5_out_path, 'w') as f:
        f.create_dataset('input_questions', data=input_questions_encoded)
        f.create_dataset('target_questions', data=target_questions_encoded)
        f.create_dataset('orig_idxs', data=np.array(orig_idxs))
        f.create_dataset('img_idxs', data=np.array(img_idxs))
        if len(question_families) > 0:
            f.create_dataset('question_families', data=np.array(question_families))
        if len(answers) > 0:
            f.create_dataset('answers', data=np.array(answers))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, required=True, help="path for CLEVR questions json files")
    parser.add_argument("-out_vocab_path", type=str, required=True, help="output path for vocab")
    parser.add_argument("-out_h5_path", type=str, required=True, help="output path for questions encoded dataset")
    parser.add_argument("-min_token_count", type=int, default=1, help="min count for adding token in vocab")

    args = parser.parse_args()

    punct_to_keep = [';', ',', '?']
    punct_to_remove = ['.']

    preprocess_questions(min_token_count=1,
                         punct_to_keep=punct_to_keep,
                         punct_to_remove=punct_to_remove,
                         add_start_token=True,
                         add_end_token=True,
                         json_data_path=args.data_path,
                         vocab_out_path=args.out_vocab_path,
                         h5_out_path=args.out_h5_path)
