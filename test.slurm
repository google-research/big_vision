#!/bin/bash
#SBATCH --job-name=bv_test   # name
#SBATCH --nodes=2                                 # nodes
#SBATCH --ntasks-per-node=1                       # crucial - only 1 task per dist per node!
#SBATCH --cpus-per-task=10                        # number of cores per tasks
#SBATCH --gres=gpu:4                              # number of gpus

#SBATCH --time 999:00:00                                # maximum execution time (HH:MM:SS)
#SBATCH --output=output/%x-%j-%a-%A.out                       # output file name
#SBATCH --exclude=compute-permanent-node-[767,98,91]

MASTER_ADDR=$(dig +short $(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1))
export MASTER_ADDR  # https://github.com/koalaman/shellcheck/wiki/SC2155
export MASTER_PORT=9092

# This also works without master port.
srun python3 test.py --master_addr $MASTER_ADDR --master_port $MASTER_PORT
