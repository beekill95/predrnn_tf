#!/bin/bash

#SBATCH -J predrnn_moving_mnist
#SBATCH -p gpu
#SBATCH -o mnist_%j.txt
#SBATCH -e mnist_%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=qmnguyen@iu.edu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task 4
#SBATCH --gpus-per-node v100:1
#SBATCH --time=24:00:00
#SBATCH --mem=64G
#SBATCH -A r00043

#Run your program
srun scripts/moving_mnist.sh
