#!/bin/bash
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_lmbonus" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt" -alpha_logits 0.2
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_lmbonus" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt" -alpha_logits 0.1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_lmbonus" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt" -alpha_logits 0.05
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_lmbonus" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt" -alpha_logits 0.5 -alpha_decay_rate 0.5
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_lmbonus" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -num_questions 8 -lm_path "output/lm_model/model.pt" -alpha_logits 0.2 -alpha_decay_rate 0.5





