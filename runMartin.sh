#!/bin/sh
#SBATCH --cpus-per-task=5
#SBATCH --job-name=testTensorFlow
#SBATCH --ntasks=1
#SBATCH --time=05:00
#SBATCH --output=slurm-%J.out
#SBATCH --gres=gpu:2
#SBATCH --constraint="V5|V6"
#SBATCH --partition=shared-gpu

module load GCC/6.4.0-2.28  OpenMPI/2.1.2 cuDNN/7.0.5-CUDA-9.1.85 foss/2016a Keras/2.1.6-Python-3.6.4 protobuf-python/3.2.0-Python-3.6.4


srun python3 Deep.py

