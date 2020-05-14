python src/train/train_Policy_SL.py -data_path "data" -out_path "output/temp"

python src/train/train_Policy_SL.py -data_path "data" -out_path "output/num_tokens_87" -word_emb_size 32 \
-hidden_size 64 -bs 512 -ep 20