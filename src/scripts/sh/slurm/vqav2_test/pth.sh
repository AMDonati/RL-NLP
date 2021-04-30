#!/bin/bash
#SBATCH --job-name=T-pth
#SBATCH --qos=qos_gpu-t4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=8g
#SBATCH --output=slurm_out/vqa/T-pth-%j.out
#SBATCH --error=slurm_out/vqa/T-pth-%j.err
#SBATCH --time=100:00:00


export TMPDIR=$JOBSCRATCH
module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp-2
export PYTHONPATH=src:${PYTHONPATH}


MODEL_PATH="output/RL/VQA_vilbert_small/proba_thr0.005"
NUM_EPISODES_TEST=20000

set -x
srun python -u src/scripts/test.py -models_path $MODEL_PATH -num_episodes_test $NUM_EPISODES_TEST -test_metrics "return" "oracle" "dialog" "bleu" "meteor" "cider" "ppl_dialog_lm" "size_valid_actions" "sum_probs" "selfbleu" "dialogimage" "language_score" "kurtosis" "peakiness"