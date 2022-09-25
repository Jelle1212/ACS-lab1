#!/bin/sh
#SBATCH --account=stud-ewi-crs-cese4010
#SBATCH --partition=cese4010
#SBATCH --qos=cese4010
#SBATCH --reservation=cese4010
#SBATCH --time=0:01:00
#SBATCH --ntasks=1
#SBATCH --gres=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=1024
#SBATCH --mail-type=FAIL

# Use this simple command to check that your sbatch settings are working
/usr/bin/scontrol show job -d "$SLURM_JOB_ID"

module use /opt/insy/modulefiles
module load cuda/11.5

srun run debug/./acsmatmult -h
