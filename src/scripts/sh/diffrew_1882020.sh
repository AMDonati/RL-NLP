#!/bin/bash
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_diffrew" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 20 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200 -diff_reward 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_diffrew" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 30 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200 -diff_reward 1
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_diffrew" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -lm_path "output/lm_model/model.pt" -num_truncated 50 -debug "0,2000" -grad_clip 1 -num_questions 8 -truncate_mode "sample_va" -num_episodes_test 200 -diff_reward 1





