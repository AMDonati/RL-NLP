#!/bin/bash
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/reinforce_2000_img_len_20" -model "lstm" -update_every 64 -agent "REINFORCE" -lr 0.0001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 20 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200 -eval_no_trunc 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/reinforce_2000_img_len_20" -model "lstm" -update_every 64 -agent "REINFORCE" -lr 0.0001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 30 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200 -eval_no_trunc 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/reinforce_2000_img_len_20" -model "lstm" -update_every 64 -agent "REINFORCE" -lr 0.0001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "proba_thr" -num_episodes_test 200 -eval_no_trunc 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/reinforce_2000_img_len_20" -model "lstm" -update_every 64 -agent "REINFORCE" -lr 0.0001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "proba_thr" -num_episodes_test 200 -p_th 0.05 -eval_no_trunc 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/reinforce_2000_img_len_20" -model "lstm" -update_every 64 -agent "REINFORCE" -lr 0.0001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 10 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "top_k" -num_episodes_test 200 -eval_no_trunc 1







