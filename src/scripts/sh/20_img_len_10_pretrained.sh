#!/usr/bin/env bash
python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10_pretrained" -model "lstm" -update_every 20 -agent "PPO" -K_epochs 10 -eps_clip 0.02 -lr 0.005 -word_emb_size 8 -hidden_size 24 -num_episodes_train 50000 -policy_path "output/RL/20_img_len_10_pretrained/experiments/pretrain/20200711-190549/model.pth" -debug "0,50"

output/lstmwordbatch_pretraining/SL_LSTM_8_24

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10_pretrained" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 8 -hidden_size 24 -num_episodes_train 50000 -policy_path "output/lstmwordbatch_pretraining/SL_LSTM_8_24/model.pt" -debug "0,50"

python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/50_img_len_10_pretrained" -model "lstm" -update_every 64 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -policy_path "output/lstmwordbatch_pretraining/SL_LSTM_32_64/model.pt" -debug "0,50"




# Telecom Paris Machine.
python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10_pretrained" -model "lstm" -update_every 64 -agent "PPO" -K_epochs 5 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -policy_path "output/SL_LSTM_32_64/model.pt" -debug "0,50"

#To launch.
python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/100_img_len_10_q1_pretrained" -model "lstm" -update_every 64 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 200000 -policy_path "output/lstmwordbatch_pretraining/SL_LSTM_32_64/model.pt" -debug "0,100" -num_questions 1