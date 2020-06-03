python src/train/train_Policy_SL.py -data_path "data" -out_path "output/temp"

python src/train/train_Policy_SL.py -data_path "data" -out_path "output" -word_emb_size 16 \
-hidden_size 32 -bs 8 -ep 50

python src/train/train_Policy_SL.py -data_path "data" -out_path "output/policy_pre_training" -word_emb_size 32 -hidden_size 64 \
 -bs 512 -ep 50

