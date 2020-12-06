#!/usr/bin/env bash
python src/scripts/run.py -env "vqa" -max_len 3 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen3" -model "lstm" -update_every 512 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,10" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "lv_norm" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 3 -min_data 1
python src/scripts/run.py -env "vqa" -max_len 3 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen3" -model "lstm" -update_every 512 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,10" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "lv_norm" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 3 -min_data 1 -p_th 0.0005
python src/scripts/run.py -env "vqa" -max_len 3 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen3" -model "lstm" -update_every 512 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,10" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "lv_norm" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 3 -min_data 1 -p_th 0.001
python src/scripts/run.py -env "vqa" -max_len 3 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen3" -model "lstm" -update_every 512 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,10" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "lv_norm" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 3 -min_data 1 -p_th 0.005
python src/scripts/run.py -env "vqa" -max_len 3 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen3" -model "lstm" -update_every 512 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 10000 -debug "0,10" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "lv_norm" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 3 -min_data 1 -p_th 0.01





