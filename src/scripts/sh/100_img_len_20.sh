#!/bin/bash
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/100_img_len_20" -model "lstm" -update_every 64 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.005 -word_emb_size 32 -hidden_size 64 -num_episodes_train 200000 -policy_path "output/lstmwordbatch_pretraining/SL_LSTM_32_64/model.pt" -debug "0,100" -num_questions 8
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/100_img_len_20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.005 -word_emb_size 32 -hidden_size 64 -num_episodes_train 200000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,100" -grad_clip 1 -num_questions 8
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/100_img_len_20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.005 -word_emb_size 32 -hidden_size 64 -num_episodes_train 200000 -num_truncated 87 -debug "0,100" -grad_clip 1 -num_questions 8
