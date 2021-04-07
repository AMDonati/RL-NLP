#!/bin/bash
#SBATCH --job-name=clevrfeatures
#SBATCH --qos=qos_gpu-t3
#SBATCH -C v100-32g
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --gres=gpu:1
#SBATCH --output=clevrfeatures%j.out
#SBATCH --error=clevrfeatures%j.err
#SBATCH --time=03:00:00

module purge
module load  pytorch-gpu/py3/1.7.1
conda activate rl-nlp

export PYTHONPATH=src:${PYTHONPATH}

set -x
srun python -u src/preprocessing/extract_features.py --input_image_dir data/CLEVR_v1.0/images/train --output_h5_file data/train_features.h5 --batch_size 64
srun python -u src/preprocessing/extract_features.py --input_image_dir data/CLEVR_v1.0/images/val --output_h5_file data/val_features.h5 --batch_size 64
srun python -u src/preprocessing/extract_features.py --input_image_dir data/CLEVR_v1.0/images/test --output_h5_file data/test_features.h5 --batch_size 64
