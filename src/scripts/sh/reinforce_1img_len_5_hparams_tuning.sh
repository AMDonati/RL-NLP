#!/bin/bash
python src/scripts/run.py -max_len 5 -data_path "data" -out_path "output/RL/reinforce_1img_len_5" -model "lstm" -update_every 1 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -eps 1e-02 -word_emb_size 8 -hidden_size 24 -num_episodes_train 50000 -debug "0,1" -num_questions 1 -entropy_coeff 0.01 -num_truncated 87
python src/scripts/run.py -max_len 5 -data_path "data" -out_path "output/RL/reinforce_1img_len_5" -model "lstm" -update_every 1 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -eps 1 -word_emb_size 8 -hidden_size 24 -num_episodes_train 50000 -debug "0,1" -num_questions 1 -entropy_coeff 0.01 -num_truncated 87
python src/scripts/run.py -max_len 5 -data_path "data" -out_path "output/RL/reinforce_1img_len_5" -model "lstm" -update_every 1 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -eps 1e-04 -word_emb_size 8 -hidden_size 24 -num_episodes_train 50000 -debug "0,1" -num_questions 1 -entropy_coeff 0.01 -num_truncated 87
python src/scripts/run.py -max_len 5 -data_path "data" -out_path "output/RL/reinforce_1img_len_5" -model "lstm" -update_every 1 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.005 -eps 1 -word_emb_size 8 -hidden_size 24 -num_episodes_train 50000 -debug "0,1" -num_questions 1 -entropy_coeff 0.01 -num_truncated 87
python src/scripts/run.py -max_len 5 -data_path "data" -out_path "output/RL/reinforce_1img_len_5" -model "lstm" -update_every 1 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.0001 -eps 1e-10 -word_emb_size 8 -hidden_size 24 -num_episodes_train 50000 -debug "0,1" -num_questions 1 -entropy_coeff 0.01 -num_truncated 87





