#!/usr/bin/env bash
#python src/scripts/test.py -models_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/best_exp/vqa_vilbert_REINFORCE_pretrain__adam_1e-05_ent0.01_epsclip0.01_graclipNone" -num_episodes_test 974 -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "language_score"
python test.py -models_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/best_exp/vqa_vilbert_PPO_proba_thr0.005_adam_5e-05_ent0.01_epsclip0.01_graclip5.0_temp2.0_div0.75_step1000_tmin0.7_tmax10.0_smax200" -num_episodes_test 974 -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "size_valid_actions" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "valid_actions" "language_score"
python test.py -models_path "output/RL/VQA_exp_smallvocab_vilbert/temperature_pth_512_1024/50000_ep/best_exp/vqa_vilbert_PPO_proba_thr0.005_adam_5e-05_ent0.02_epsclip0.01_graclip5.0_temp2.0_div0.75_step1000_tmin0.7_tmax10.0_smax200" -num_episodes_test 974 -test_metrics "return" "dialog" "bleu" "ppl_dialog_lm" "size_valid_actions" "ttr_question" "sum_probs" "ppl" "lv_norm" "ttr" "selfbleu" "dialogimage" "valid_actions" "language_score"