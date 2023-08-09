#!/bin/bash

# set up job
#SBATCH --qos=regular
#SBATCH --time=15:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=128
#SBATCH --constraint=cpu
#SBATCH --job-name=oscillations
#SBATCH --output=Mn-4Tesla.out

# run python
module load python
echo "SLURM: start oscillations.py"
date
python oscillations.py
echo "SLURM: finish oscillations.py"
date
