python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_train_questions.json" \
-out_vocab_path "data/vocab.json" -out_h5_path "data/train_questions.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_val_questions.json" \
-out_vocab_path "data/vocab.json" -out_h5_path "data/val_questions.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_test_questions.json" \
-out_vocab_path "data/vocab.json" -out_h5_path "data/test_questions.h5" -min_token_count 1

