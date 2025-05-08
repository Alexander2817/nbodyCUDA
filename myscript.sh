#!/bin/bash
#SBATCH --job-name=nbody_simulation
#SBATCH --partition=GPU
#SBATCH --time=00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1

module load cuda/12.4

echo "Running with 1000 particles..."
./nbody_cuda 1000 0.01 10 10 128 > output_1000_particles.txt

echo "Running with 10000 particles..."
./nbody_cuda 10000 0.01 10 10 128 > output_10000_particles.txt

echo "Running with 100000 particles..."
./nbody_cuda 100000 0.01 10 10 128 > output_100000_particles.txt
