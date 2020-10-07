#!/bin/bash
#python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/20000_img_len_40_debug" -model "lstm" -update_every 128 -agent "REINFORCE" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 200 -train_seed 1 -num_questions 8 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/policy_pretraining/LSTMBatch_32_64_VQA/model.pt"
#python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/20000_img_len_40_debug" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/lm_model/model.pt" -condition_answer "after_fusion" -num_episodes_test 200 -train_seed 1 -num_questions 1 -mask_answers 1 -policy_path "output/policy_pretraining/LSTMBatch_32_64_VQA/model.pt"
python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/20000_img_len_40_debug" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 200 -train_seed 1 -num_questions 8 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/policy_pretraining/LSTMBatch_32_64_VQA/model.pt" -truncate_mode "top_k" -num_truncated 30
python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/20000_img_len_40_debug" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 200 -train_seed 1 -num_questions 8 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/policy_pretraining/LSTMBatch_32_64_VQA/model.pt" -truncate_mode "top_k" -num_truncated 40
python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/20000_img_len_40_debug" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 200 -train_seed 1 -num_questions 8 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/policy_pretraining/LSTMBatch_32_64_VQA/model.pt" -truncate_mode "top_k" -num_truncated 50
python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/20000_img_len_40_debug" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 200 -train_seed 1 -num_questions 8 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/policy_pretraining/LSTMBatch_32_64_VQA/model.pt" -truncate_mode "top_k" -num_truncated 20
python src/scripts/run.py -max_len 40 -data_path "data" -out_path "output/RL/20000_img_len_40_debug" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/lm_model/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 200 -train_seed 1 -num_questions 8 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -policy_path "output/policy_pretraining/LSTMBatch_32_64_VQA/model.pt" -truncate_mode "top_k" -num_truncated 86

