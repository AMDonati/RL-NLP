#!/usr/bin/env bash
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth_512_1024/hparams" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.0001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 20000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.004 -s_min 1 -s_max 200 -entropy_coeff 0.01
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth_512_1024/hparams" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.0001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 20000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.004 -s_min 1 -s_max 200 -entropy_coeff 0.01 -diff_reward 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth_512_1024/hparams" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.0001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 20000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.004 -s_min 1 -s_max 200 -entropy_coeff 0.001
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth_512_1024/hparams" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.0001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 20000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.004 -s_min 1 -s_max 200 -entropy_coeff 0.001 -diff_reward 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth_512_1024/hparams" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.001 -lr 0.0001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 20000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.004 -s_min 1 -s_max 200 -entropy_coeff 0.01
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth_512_1024/hparams" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.001 -lr 0.0001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 20000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.004 -s_min 1 -s_max 200 -entropy_coeff 0.01 -diff_reward 1
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth_512_1024/hparams" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 20000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.004 -s_min 1 -s_max 200 -entropy_coeff 0.01
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_pth_512_1024/hparams" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.01 -lr 0.00005 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 20000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "proba_thr" -grad_clip 5 -fusion "average" -condition_answer "after_fusion" -device_id 2 -min_data 1 -temperature 2 -temp_step 1000 -temp_factor 0.75 -temp_min 0.7 -p_th 0.004 -s_min 1 -s_max 200 -entropy_coeff 0.01 -diff_reward 1