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
#	 sbatch --account=bibs-frankmj-condo hist.sh
# To use new cluster:
# 	 sbatch --account=carney-frankmj-condo hist.sh
#
# Once the job starts you will see a file MySerialJob-****.out
# The **** will be the slurm JobID

# --- Start of slurm commands -----------

#SBATCH --account=carney-frankmj-condo

# Request specified runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.
# Use more memory (4GB):
##SBATCH --mem=4G

# Array ranges
#SBATCH --array=50,100,250,500

# Specify a job name:
#SBATCH -J HistR

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o ./output/HistR-%j.out
#SBATCH -e ./output/HistR-%j.err

#----- End of slurm commands ----
# env norm k reprocess comb
echo "Starting job $SLURM_ARRAY_TASK_ID"
# 					  env     
# python3 hist_grid.py "80"	 500 	5000 	$SLURM_ARRAY_TASK_ID
python3 complexity_hist.py "80"	 500 	5000 	$SLURM_ARRAY_TASK_ID
# python3 opt_parameters.py $SLURM_ARRAY_TASK_ID

