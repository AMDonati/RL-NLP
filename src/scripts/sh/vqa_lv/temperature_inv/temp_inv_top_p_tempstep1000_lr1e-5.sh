#!/usr/bin/env bash
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.00001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "top_p" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 0 -min_data 1 -temperature 1 -temp_step 1000 -temp_factor 1.1 -temp_min 0.5 -temp_max 1.5 -top_p 0.9 -s_min 10 -s_max 200
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.00001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "top_p" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 0 -min_data 1 -temperature 1 -temp_step 1000 -temp_factor 1.1 -temp_min 0.5 -temp_max 1.5 -top_p 0.9 -s_min 10 -s_max 200 -entropy_coeff 0.1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.00001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "top_p" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 0 -min_data 1 -temperature 1 -temp_step 1000 -temp_factor 1.1 -temp_min 0.5 -temp_max 1.5 -top_p 0.9 -s_min 10 -s_max 200 -entropy_coeff 0.5

