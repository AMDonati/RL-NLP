#!/bin/bash
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_vqa" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 87 -debug "0,2000" -grad_clip 1 -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 200 -truncate_mode "proba_thr" -p_th 0.05 -train_seed 1 -num_questions 8 -reward_vocab "data/closure_vocab.json"
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_vqa" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 20 -debug "0,2000" -grad_clip 1 -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 200 -truncate_mode "sample_va" -train_seed 1 -num_questions 8 -reward_vocab "data/closure_vocab.json"
python src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2000_img_len_20_vqa" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -num_truncated 10 -debug "0,2000" -grad_clip 1 -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 200 -truncate_mode "top_k" -train_seed 1 -num_questions 8 -reward_vocab "data/closure_vocab.json"

