python src/train/train_LM_network.py -model "ln_lstm" -num_layers 1 -emb_size 256 \
-hidden_size 512 -p_drop 0.1 -grad_clip 1 -data_path "data" \
-out_path "output" -lr 0.0005 -bs 512 -ep 30 -num_workers 5

# to train on a subset of question samples.
python src/train/train_LM_network.py -num_layers 1 -emb_size 64 \
-hidden_size 128 -p_drop 0 -grad_clip False -data_path "data/CLEVR_v1.0/temp/50000_20000_samples" \
-out_path "output" -cuda True -bs 128 -ep 30 -skip_training False -num_workers 5 - eval False