if __name__ == '__main__':
    from transformers import BertTokenizer, GPT2Tokenizer
    import argparse
    from data_provider.vqa_tokenizer import VQATokenizer
    import numpy as np
    from data_provider._image_features_reader import ImageFeaturesH5Reader
    from data_provider.vqa_dataset import VQADataset

    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", type=str, default='data/vqa-v2',
                        help="data folder containing questions embeddings and img features")
    parser.add_argument("-features_path", type=str, default="data/vqa-v2/coco_trainval.lmdb",
                        help="data folder containing questions embeddings and img features")
    parser.add_argument("-vocab_path", type=str, default="none")
    parser.add_argument("-split", type=str, default="train", help="train or val splits.")
    parser.add_argument("-min_split", type=int, default=0,
                        help="if true, build a min+split dataset; or a minimal vocab if vocab_path is none")
    parser.add_argument("-test", type=int, default=1)
    args = parser.parse_args()

    lm_tokenizer = GPT2Tokenizer.from_pretrained("cache/gpt-2")
    reward_tokenizer = BertTokenizer.from_pretrained('cache/bert')
    features_h5path = args.features_path
    images_feature_reader = ImageFeaturesH5Reader(features_h5path, False)
    question_tokenizer = VQATokenizer(lm_tokenizer=lm_tokenizer)

    if args.vocab_path == "none":
        split = "trainval" if not args.min_split else "mintrainval"
        vqa_dataset = VQADataset(split=split, dataroot=args.data_path,
                                 vocab_path=args.vocab_path,
                                 question_tokenizer=question_tokenizer, image_features_reader=images_feature_reader,
                                 reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                                 num_images=None)

    else:
        split = args.split if not args.min_split else "min" + args.split
        print("Building {} dataset from {}...".format(split, args.vocab_path))
        vqa_dataset = VQADataset(split=split, dataroot=args.data_path,
                                 question_tokenizer=question_tokenizer, image_features_reader=images_feature_reader,
                                 reward_tokenizer=reward_tokenizer, clean_datasets=True, max_seq_length=23,
                                 num_images=None, vocab_path=args.vocab_path)

        if args.test:
            vocab = vqa_dataset.vocab_questions
            new_d = {}
            for k in sorted(vocab, key=len):
                new_d[k] = vocab[k]

            # test of answers vocab:
            answers_ids = list(vqa_dataset.ans2label.values())
            print("first id", answers_ids[0])
            print("last id", answers_ids[-1])
            print("len vocab answers", vqa_dataset.len_vocab_answer)

            # test of translate functions:
            print("Test of reward tokenizer...")
            print('Is there a pizza?')
            lm_idx = vqa_dataset.lm_tokenizer.encode('Is there a pizza?')
            input_idx = [vqa_dataset.lm_to_dataset_trad[idx] for idx in lm_idx]
            reward_idx = vqa_dataset.translate_for_reward(input_idx)
            question_decoded = vqa_dataset.reward_tokenizer.decode(reward_idx)
            print('question decoded', question_decoded)

            print("Test of lm_to_dataset_function ...")
            idx = np.random.randint(vqa_dataset.len_vocab)
            token_idx = list(vqa_dataset.lm_to_dataset_trad.keys())[idx]
            print("word from lm_tokenizer")
            print(vqa_dataset.lm_tokenizer.decoder[token_idx])
            trad_token_idx = vqa_dataset.lm_to_dataset_trad[token_idx]
            print("word from dataset vocab")
            print(vqa_dataset.question_tokenizer.decode([trad_token_idx]))

            print("test of get_item function...")
            (inputs, targets), labels, (features, image_mask, spatials) = vqa_dataset.__getitem__(1)
            print("inputs", inputs)
            print("targets", targets)
            print("answer labels", labels.shape)
            print("features", features.shape)
            print("image_mask", image_mask.shape)
            print("spatials", spatials.shape)

            print("test of get_data_for_VILBERT function...")
            features, spatials, image_mask, question, target, input_mask, segment_ids, co_attention_mask, question_id = vqa_dataset.get_data_for_ViLBERT(
                index=0)
            print("question", question)
            print("target", target.shape)  # 3129 answers.

            print("print test of decode function...")
            entry = vqa_dataset.filtered_entries[0]
            print("true question:{}".format(entry["question"]))
            print("question decoded - question_tokenizer: {}".format(
                vqa_dataset.question_tokenizer.decode(entry["q_token"].numpy())))
            print("question decoded - lm_tokenizer: {}".format(
                vqa_dataset.lm_tokenizer.decode(entry["q_token_lm"].numpy())))
