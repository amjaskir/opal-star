#!/bin/bash

# This is an example batch script for slurm on Oscar
# 
# The commands for slurm start with #SBATCH
# All slurm commands need to come before the program 
# you want to run.
#
# This is a bash script, so any line that starts with # is
# a comment.  If you need to comment out an #SBATCH line 
# use ##SBATCH 
#
# To submit this script to slurm do:
#    sbatch hist.sh
# To use frank lab condo use:
#	 sbatch --account=bibs-frankmj-condo compare.sh
#
# Once the job starts you will see a file MySerialJob-****.out
# The **** will be the slurm JobID

# --- Start of slurm commands -----------

# Request specified runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.
# Use more memory (4GB):
##SBATCH --mem=4G	

# Specify a job name:
#SBATCH -J Compare

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o ./output_grid/Compare-%j.out
#SBATCH -e ./output_grid/Compare-%j.err

#----- End of slurm commands ----
# env norm k reprocess comb
echo "Starting job $SLURM_ARRAY_TASK_ID"
# 					  		env    reprocess 
# compare effect of normalization on each env
# python3 compare_norm.py "lean" 1 
# python3 compare_norm.py "rich" 1 
# python3 compare_norm.py "rich_90_80_70" 1
# python3 compare_norm.py "lean_30_20_10" 1

# python3 compare_norm_bayes.py "lean" 1 
# python3 compare_norm_bayes.py "rich" 1 
# python3 compare_norm_bayes.py "rich_90_80_70" 1
# python3 compare_norm_bayes.py "lean_30_20_10" 1

# version reprocess
# compare whether model performance on rich/lean
python3 compare_env.py "std" 1 
python3 compare_env.py "bayes" 1 
