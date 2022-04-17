#!/bin/bash
#SBATCH --nodes=1
#SBATCH --mem=5G
#SBATCH --ntasks=4
#SBATCH --gpus-per-node=1
#SBATCH --time=24:00:00
#SBATCH --output=/home/mjyothi/scratch/one4all/run1/%x-%j-log.out

module load gcc/9.3.0 arrow cuda/11 python/3.8
source /home/mjyothi/home/mjyothi/bart/bin/activate
python -m experiment.one4all.exp01 -o /home/mjyothi/scratch/heirarchical/run1 --cache_dir /home/mjyothi/home/mjyothi/cache --initial_wts_dir /home/mjyothi/home/mjyothi/initial_wts &> /home/mjyothi/scratch/one4all/run1/run.log
