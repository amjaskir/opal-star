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
#	 sbatch --account=bibs-frankmj-condo summary.sh
# To use new cluster:
# 	 sbatch --account=carney-frankmj-condo summary.sh
#
# Once the job starts you will see a file MySerialJob-****.out
# The **** will be the slurm JobID

# --- Start of slurm commands -----------

# Run on Carney cluster
#SBATCH --account=carney-frankmj-condo

# Request specified runtime:
#SBATCH --time=0:15:00

# Default resources are 1 core with 2.8GB of memory.
# Use more memory (4GB):
#SBATCH --mem=13G

# Array ranges
#SBATCH --array=50,100,250,500

# Specify a job name:
#SBATCH -J SummaryL

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o ./output_grid/SummaryL-%j.out
#SBATCH -e ./output_grid/SummaryL-%j.err

#----- End of slurm commands ----
# env norm k reprocess comb
echo "Starting job $SLURM_ARRAY_TASK_ID"

# python3 summary_hist.py "[1,20,200]" 100 	"rich"
python3 summary_hist.py "[10,20]" $SLURM_ARRAY_TASK_ID 	"rich"
