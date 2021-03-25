#!/usr/bin/env bash
VQA_LM_MODEL=1lYv4jpaITS7DEPn2ZgH7I-jTfy7CqeRz
VQA_LM_MODEL_SMALLVOCAB=1u8rv6z31cSU9Yt4AzHfI4esXFmU0TcXB
VQA_POLICY_512_1024_ANSWER=1_QcidAIzKVsyqzzliHHaW7AOZH1oE0Ei
VQA_POLICY_512_1024_ANSWER_SMALLVOCAB=1m75UmBajV7q7OFanGkHgg7YmqekWuWF_
BERT_BASE_CONFIG_FILE=1OtxCDSqE1mTgSoeBpJSbGNcbN0Pc81bx

mkdir output/old

mkdir output/vqa_lm_model
mkdir output/vqa_lm_model_smallvocab
mkdir output/vqa_policy_512_1024_answer
mkdir output/vqa_policy_512_1024_answer_smallvocab
mkdir output/vilbert_vqav2

mv output/vqa_lm_model/model.pt output/old/vqa_lm_model_$(date +"%Y_%m_%d_%I_%M_%p")
gdown --id $VQA_LM_MODEL --output output/vqa_lm_model/model.pt

mv output/vqa_lm_model_vocab/model.pt output/old/vqa_lm_model_smallvocab_$(date +"%Y_%m_%d_%I_%M_%p")
gdown --id $VQA_LM_MODEL_SMALLVOCAB --output output/vqa_lm_model_smallvocab/model.pt

mv output/vqa_policy_512_1024_answer/model.pt output/old/vqa_policy_512_1024_answer_$(date +"%Y_%m_%d_%I_%M_%p")
gdown --id $VQA_POLICY_512_1024_ANSWER --output output/vqa_policy_512_1024_answer/model.pt

mv output/vqa_policy_512_1024_answer_smallvocab/model.pt output/old/vqa_policy_512_1024_answer_smallvocab_$(date +"%Y_%m_%d_%I_%M_%p")
gdown --id $VQA_POLICY_512_1024_ANSWER_SMALLVOCAB --output output/vqa_policy_512_1024_answer_smallvocab/model.pt

gdown --id $BERT_BASE_CONFIG_FILE --output output/vilbert_vqav2/bert_base_6layer_6conect.json