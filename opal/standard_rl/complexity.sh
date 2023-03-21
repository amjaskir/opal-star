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
#    sbatch grid.search
# To use frank lab condo use:
#	 sbatch --account=bibs-frankmj-condo complexity.sh
# To use new cluster:
# 	 sbatch --account=carney-frankmj-condo complexity.sh
#
# Once the job starts you will see a file MySerialJob-****.out
# The **** will be the slurm JobID

# --- Start of slurm commands -----------

# use the frank carney clusters
#SBATCH --account=carney-frankmj-condo

# Request specified runtime:
#SBATCH --time=25:00:00
# --time=25:00:00
# should be about 4 hours per complexity level

# Default resources are 1 core with 2.8GB of memory.
# Use more memory (4GB):
##SBATCH --mem=4G

# MaxArraySize is 10001 so cannot exceed 10000
# maximum number of combinations for grid_search.py is 800
# Upperbound should = arg6 - 1     
#SBATCH --array=0-99                       ################################## Make sure this is updated

# Specify a job name:
#SBATCH -J ComplexityRL_L

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o ./output/ComplexityRL_L%j.out
#SBATCH -e ./output/ComplexityRL_L%j.err	

#----- End of slurm commands ----

echo "Starting job $SLURM_ARRAY_TASK_ID"
# python3 complexity.py $SLURM_ARRAY_TASK_ID "rich"		500 		5000 		100
python3 complexity.py $SLURM_ARRAY_TASK_ID "lean"		500 		5000 		100	