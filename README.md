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
#### Link to the pre-trained models 
* Language Model .pt file [here](https://drive.google.com/drive/folders/1zRT4EF8xNmilzZMYysyhCj73oQKvBLsX?usp=sharing). 
* Pretrained Policy .pt file (word_emb_size = 32, hidden_size = 64) [here](https://drive.google.com/file/d/1m_pXVQwQ41jgDUwuBvRHJ1U-GLqKRd3N/view?usp=sharing). 
### Training the Language Model on the Dataset of Questions
`python src/train/train_LM_network.py -model "lstm" -num_layers 1 -emb_size 32  \`
`-hidden_size 64 -p_drop 0 -data_path "data" \`
`-out_path "output" -bs 512 -ep 20 -num_workers 0`

### Pre-training of the Policy with Supervised Learning 
`python src/train/train_Policy_SL.py -data_path "data" -out_path "output/policy_pre_training" -word_emb_size 32 -hidden_size 64 \
 -bs 512 -ep 50 -num_workers 0 -max_samples 21`  
 N.B: When training only on a CPU, the max_samples args is required to train only on a subset of the dataset. 
 
### Training the RL Agent 
#### from scrach (arg -truncate_mode is None): ex on 2000 Img, length of episode = 20, 50,000 episodes. 
`python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt"`
#### with policy pretraining: add the -policy_path arg. 
`python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -policy_path "output/lstmwordbatch_pretraining/SL_LSTM_32_64/model.pt" -debug "0,2000" -num_questions 8 -num_truncated 87 -lm_path "output/lm_model/model.pt"`
#### with truncation: add the -truncate_mode arg (choose between 'top_k', 'proba_thr', 'sample_va')
* `python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "top_k"`
* `python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_truncate_modes" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "proba_thr" -num_episodes_test 200`
* `python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_truncate_modes" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 30 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200`
#### Eval only (after training): -num_episodes_train 0 and -policy_path arg is used to upload the trained policy model. 
`python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 0 -num_truncated 87 -debug "0,2000" -num_questions 8 -lm_path "output/lm_model/model.pt" -policy_path "output/RL/2000_img_len_20/experiments/train/02/model.pth" -num_episodes_test 200`

#### logging on tensorboard to display results: 
* `cd output/2000_img_len_20"`
* `tensorboard --logdir=experiments/train`
