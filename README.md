# RL-NLP

## CLEVR Datasets: 
* [CLEVR original dataset page](https://cs.stanford.edu/people/jcjohns/clevr/)
* [CLEVR preprocessing github](https://github.com/facebookresearch/clevr-iep/blob/master/TRAINING.md)
* [CLOSURE dataset github page](https://github.com/rizar/CLOSURE)

## Downloading the dataset
The CLEVR Dataset can be downloaded [here](https://cs.stanford.edu/people/jcjohns/clevr/).   
The full dataset (questions + images) or only the questions part can be downloaded. 
To avoid memory issues on local machines, the first 21 images of the train/val/test datasets are available for download [here](https://drive.google.com/drive/folders/1OEy8Dfq2mO-vAiL9hFO1E_HbqC0wX4WB?usp=sharing).

## Data preprocessing
* To run all the scripts from the origin repo (RL-NLP), run first the following command line: `export PYTHONPATH=src:${PYTHONPATH}`

### Preprocessing the dataset questions
To preprocess the questions of the three datasets, run the 3 following command lines (in this order): 

* `python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_train_questions.json" \`
`-out_vocab_path "data/vocab.json" -out_h5_path "data/train_questions.h5" -min_token_count 1`

* `python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_val_questions.json" \`
`-out_vocab_path "data/vocab.json" -out_h5_path "data/val_questions.h5" -min_token_count 1`

* `python src/preprocessing/preprocess_questions.py -data_path "data/CLEVR_v1.0/questions/CLEVR_test_questions.json" \`
`-out_vocab_path "data/vocab.json" -out_h5_path "data/test_questions.h5" -min_token_count 1`

### Extracting the image features
To extract the image features, run the following command lines (batch size arg must be tuned depending on memory availability): 
* `python src/preprocessing/extract_features.py \`
  `--input_image_dir data/CLEVR_v1.0/images/train \`
  `--output_h5_file data/train_features.h5 --batch_size 128`

* `python src/preprocessing/extract_features.py \`
  `--input_image_dir data/CLEVR_v1.0/images/val \`
  `--output_h5_file data/val_features.h5 --batch_size 128`

* `python src/preprocessing/extract_features.py \`
  `--input_image_dir data/CLEVR_v1.0/images/test \`
  `--output_h5_file data/test_features.h5 --batch_size 128`

## Training the models 
### Training the Language Model on the Dataset of Questions
`python src/train/train_LM_network.py -model "lstm" -num_layers 1 -emb_size 32  \`
`-hidden_size 64 -p_drop 0 -data_path "data" \`
`-out_path "output" -bs 512 -ep 20 -num_workers 0`

### Pre-training of the Policy with Supervised Learning 
`python src/train/train_Policy_SL.py -data_path "data" -out_path "output/policy_pre_training" -word_emb_size 32 -hidden_size 64 \
 -bs 512 -ep 50 -num_workers 0 -max_samples 21`  
 N.B: When training only on a CPU, the max_samples args is required to train only on a subset of the dataset. 
 
### Training the RL Agent 
`python src/scripts/run.py -max_len 10 -data_path "data" \`
`-out_path "output/RL/ppo_debug_len_10" \`
`-model "lstm" -update_every 20 \`
`-debug 1 \`
`-agent "PPO" -K_epochs 10 -eps_clip 0.02 -word_emb_size 8 -hidden_size 24`

#### logging on tensorboard to display results: 
* `cd output/ppo_debug_len_10"`
* `tensorboard --logdir=experiments/train`
