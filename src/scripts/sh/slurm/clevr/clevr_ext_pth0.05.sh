#!/bin/bash
#SBATCH --job-name=clevr-pth0.05
#SBATCH --qos=qos_gpu-t3
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --output=slurm_out/clevr/pth0.05-%j.out
#SBATCH --error=slurm_out/clevr/pth0.05-%j.err
#SBATCH --time=20:00:00

export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2

export PYTHONPATH=src:${PYTHONPATH}

set -x
srun python -u src/scripts/run.py -max_len 20 -data_path "data" -out_path "output/RL/2021_exp_clvr_vqa_20000img_len20" -model "lstm" -update_every 128 -agent "PPO" -K_epochs 20 -eps_clip 0.02 -lr 0.001 -word_emb_size 32 -hidden_size 64 -num_episodes_train 50000 -debug "0,20000" -lm_path "output/lm_ext/model.pt" -reward "vqa" -reward_path "output/vqa_model_film/model.pt" -condition_answer "after_fusion" -num_episodes_test 5000 -reward_vocab "data/closure_vocab.json" -mask_answers 1 -truncate_mode "proba_thr" -p_th 0.05 -grad_clip 1

