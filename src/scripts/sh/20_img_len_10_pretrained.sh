#!/usr/bin/env bash
python src/scripts/run.py -max_len 10 -data_path "data" -out_path "output/RL/20_img_len_10_pretrained" -model "lstm" -update_every 20 -agent "PPO" -K_epochs 10 -eps_clip 0.02 -lr 0.005 -word_emb_size 8 -hidden_size 24 -num_episodes_train 50000 -policy_path "output/RL/20_img_len_10_pretrained/experiments/pretrain/20200711-190549/model.pth" -debug "0,50"


