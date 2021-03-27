#!/usr/bin/env bash
#echo "---- k = 20-------"
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 500 -temp_factor 0.75 -temp_min 1 -num_truncated 20
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 250 -temp_factor 0.75 -temp_min 1 -num_truncated 20
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 1000 -temp_factor 0.75 -temp_min 1 -num_truncated 20
#echo "---- k = 50 -------"
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 500 -temp_factor 0.75 -temp_min 1 -num_truncated 50
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 250 -temp_factor 0.75 -temp_min 1 -num_truncated 50
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 1000 -temp_factor 0.75 -temp_min 1 -num_truncated 50
echo "---- k = 100 -------"
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 500 -temp_factor 0.75 -temp_min 1 -num_truncated 100
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 250 -temp_factor 0.75 -temp_min 1 -num_truncated 100
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 1000 -temp_factor 0.75 -temp_min 1 -num_truncated 100
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 2500 -temp_factor 0.75 -temp_min 1 -num_truncated 100
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 5000 -temp_factor 0.75 -temp_min 1 -num_truncated 100
python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 10000 -temp_factor 0.75 -temp_min 1 -num_truncated 100
#echo "---- k = 250 -------"
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 500 -temp_factor 0.75 -temp_min 1 -num_truncated 250
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 250 -temp_factor 0.75 -temp_min 1 -num_truncated 250
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 1000 -temp_factor 0.75 -temp_min 1 -num_truncated 250
#echo "---- k = 500 -------"
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 500 -temp_factor 0.75 -temp_min 1 -num_truncated 500
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 250 -temp_factor 0.75 -temp_min 1 -num_truncated 500
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 1000 -temp_factor 0.75 -temp_min 1 -num_truncated 500
#echo "---- k = 750 -------"
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 500 -temp_factor 0.75 -temp_min 1 -num_truncated 750
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 250 -temp_factor 0.75 -temp_min 1 -num_truncated 750
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 1000 -temp_factor 0.75 -temp_min 1 -num_truncated 750
#echo "---- k = 1000 -------"
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 500 -temp_factor 0.75 -temp_min 1 -num_truncated 1000
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 250 -temp_factor 0.75 -temp_min 1 -num_truncated 1000
#python src/scripts/run.py -env "vqa" -max_len 10 -data_path "data/vqa-v2" -out_path "output/RL/VQA_exp_size_truncation_smallvocab_maxlen10/temperature_2" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 512 -hidden_size 1024 -num_episodes_train 30000 -debug "0,2000" -lm_path "output/vqa_lm_model_smallvocab/model.pt" -reward "bleu_sf2" -num_episodes_test 100 -mask_answers 1 -truncate_mode "sample_va" -grad_clip 1 -fusion "average" -condition_answer "after_fusion" -init_text "Here are a few examples:" -custom_init 100 -device_id 2 -min_data 1 -temperature 5 -temp_step 1000 -temp_factor 0.75 -temp_min 1 -num_truncated 1000