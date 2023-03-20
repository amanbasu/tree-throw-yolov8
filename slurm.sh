#!/bin/bash

#SBATCH -J box_75
#SBATCH -p gpu
#SBATCH -A r00060
#SBATCH -e logs/filename_%j.err
#SBATCH -o logs/filename_%j.txt
#SBATCH --mail-type=ALL
#SBATCH --mail-user=amanagar@iu.edu
#SBATCH --gpus-per-node=v100:1
#SBATCH --nodes=1
#SBATCH --time=40:00:00
#SBATCH --mem=196G

#Load any modules that your program needs
module load anaconda/python3.8

#Run your program
python mytrain.py
# python train.py --img 400 --batch 64 --epochs 300 --data ../FADS_EAS_Tree-Throw-Prediction/datasets5/TreeThrow.yaml --name all_data --augment --cache --resume