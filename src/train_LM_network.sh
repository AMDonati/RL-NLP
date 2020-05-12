# the folder data needs to contain the following files:
* "train_questions.h5"
* "val_questions.h5"
* "test_questions.h5"
$ "vocab.json"

python src/train/train_LM_network.py -model "lstm" -num_layers 2 -emb_size 512  \
-hidden_size 512 -p_drop 0.1 -data_path "data" \
-out_path "output" -bs 512 -ep 20 -num_workers 5

# to train on a subset of question samples.
python src/train/train_LM_network.py -num_layers 1 -emb_size 16 \
-hidden_size 32 -p_drop 0  -data_path "data/CLEVR_v1.0/temp/50000_20000_samples" \
-out_path "output/temp" -bs 128 -ep 20