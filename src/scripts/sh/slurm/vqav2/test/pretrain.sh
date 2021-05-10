#!/bin/bash
#SBATCH --job-name=pretrain
#SBATCH --qos=qos_gpu-t4
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/vqa_test/pretrain%j.out
#SBATCH --error=slurm_out/vqa_test/pretrain%j.err
#SBATCH --time=100:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2

export PYTHONPATH=src:${PYTHONPATH}

POLICY_PATH="output/RL/VQAv2/vqa_vilbert_rank2_PPO_answ-img_sampling_pretrain__adam_1e-05_ent0.01_epsclip0.01_graclipNone/20210428-114034/checkpoints/model.pt"
OUT_PATH=output/RL/VQAv2/pretrain_test/

srun python src/scripts/run.py -env vqa -max_len 10 -data_path data/vqa-v2/ -out_path $OUT_PATH -model lstm -update_every 128 -agent PPO -K_epochs 20 -eps_clip 0.01 -lr 0.00001 -word_emb_size 128 -hidden_size 256 -num_episodes_train 0 -lm_path output/vqa_lm_model/model.pt -reward vilbert_rank2 -num_episodes_test 20000 -mask_answers 1 -fusion average -condition_answer after_fusion -features_path data/vqa-v2/coco_trainval.lmdb/ -reward_vocab output/vilbert_vqav2/bert_base_6layer_6conect.json -reward_path output/vilbert_vqav2/model.bin -policy_path $POLICY_PATH