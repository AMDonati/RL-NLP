python src/train/train_LM_network.py -num_layers 1 -emb_size 256 \
-hidden_size 512 -p_drop 0 -grad_clip False -data_path "data" \
-out_path "output" -cuda True

data/CLEVR_v1.0/temp/50000_20000_samples

python src/train/train_LM_network.py -num_layers 1 -emb_size 64 \
-hidden_size 128 -p_drop 0 -grad_clip False -data_path "data/CLEVR_v1.0/temp/50000_20000_samples" \
-out_path "output" -cuda True