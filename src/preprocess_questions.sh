python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_train_questions.json" \
-out_vocab_path "data/vocab.json" -out_h5_path "data/train_questions.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_val_questions.json" \
-out_vocab_path "data/vocab.json" -out_h5_path "data/val_questions.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_test_questions.json" \
-out_vocab_path "data/vocab.json" -out_h5_path "data/test_questions.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/temp/train_questions_50000_samples.json" \
-out_vocab_path "data/CLEVR_v1.0/temp/vocab.json" -out_h5_path "data/CLEVR_v1.0/temp/train_questions_50000_samples.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/temp/val_questions_20000_samples.json" \
-out_vocab_path "data/CLEVR_v1.0/temp/vocab.json" -out_h5_path "data/CLEVR_v1.0/temp/val_questions_20000_samples.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/temp/test_questions_20000_samples.json" \
-out_vocab_path "data/CLEVR_v1.0/temp/vocab.json" -out_h5_path "data/CLEVR_v1.0/temp/test_questions_20000_samples.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/temp/50000_20000_samples/train_questions.json" \
-out_vocab_path "data/CLEVR_v1.0/temp/50000_20000_samples/vocab.json" -out_h5_path "data/CLEVR_v1.0/temp/50000_20000_samples/train_questions.h5" -min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/temp/50000_20000_samples/val_questions.json" \
-out_vocab_path "data/CLEVR_v1.0/temp/50000_20000_samples/vocab.json" -out_h5_path "data/CLEVR_v1.0/temp/50000_20000_samples/val_questions.h5" \
-min_token_count 1

python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/temp/50000_20000_samples/test_questions.json" \
-out_vocab_path "data/CLEVR_v1.0/temp/50000_20000_samples/vocab.json" -out_h5_path "data/CLEVR_v1.0/temp/50000_20000_samples/test_questions.h5" \
-min_token_count 1