#!/bin/sh
#SBATCH --cpus-per-task=3
#SBATCH --job-name=testTensorFlow
#SBATCH --ntasks=1
#SBATCH --time=12:00
#SBATCH --output=slurm-%J.out
#SBATCH --gres=gpu:1
#SBATCH --partition=shared-gpu

module load foss/2018a Keras/2.1.6-Python-3.6.4 cuDNN/7.0.5-CUDA-9.1.85 CUDA TensorFlow/1.7.0-Python-3.6.4 CUDA/9.1.85 h5py/2.7.1-Python-3.6.4

srun python3 Deep.py

