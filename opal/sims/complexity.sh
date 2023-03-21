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
#SBATCH --time=3:00:00
# should be about 1 hour per complexity level

# Default resources are 1 core with 2.8GB of memory.
# Use more memory (4GB):
##SBATCH --mem=4G

# MaxArraySize is 10001 so cannot exceed 10000
# maximum number of combinations for grid_search.py is 800
# Upperbound should = arg6 - 1              

################################## Make sure this is updated
##SBATCH --array=0-207                       
##SBATCH --array=50,100,250,500,1000
#SBATCH --array=1-200                      # for UCB grid

# Specify a job name:
#SBATCH -J complexBMOD

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o ./output_grid/complexBMOD_%j.out
#SBATCH -e ./output_grid/complexBMOD_%j.err	

#----- End of slurm commands ---- 

# echo "Starting job $SLURM_ARRAY_TASK_ID"

python3 complexity_UCB_grid.py $SLURM_ARRAY_TASK_ID     "80"    1000    1000
python3 complexity_UCB_grid.py $SLURM_ARRAY_TASK_ID     "30"    1000    1000

# python3 complexity_UCB.py "80"		$SLURM_ARRAY_TASK_ID    1000 
# python3 complexity_UCB.py "30"		$SLURM_ARRAY_TASK_ID    1000 

# python3 complexity_rl.py $SLURM_ARRAY_TASK_ID "80"		1000    1000     208 
# python3 complexity_rl.py $SLURM_ARRAY_TASK_ID "30"		1000    1000     208 

# Bayes needs ~1hr, RL needs 1.5
# python3 complexity_revisions.py $SLURM_ARRAY_TASK_ID "80"		1000    1000     208  "SA" 1 0 10 0 "bmod"
# python3 complexity_revisions.py $SLURM_ARRAY_TASK_ID "30"		1000    1000     208  "SA" 1 0 10 0 "bmod"










#											env     trials  states  n_params variant
# python3 complexity.py $SLURM_ARRAY_TASK_ID "30"		200 	5000 	180    
# python3 complexity_revisions.py $SLURM_ARRAY_TASK_ID "80"		1000    1000     416  "Bayes-SA" 1 0 10 0 # Bayes needs ~1hr, RL needs 1.5
# python3 complexity_revisions.py $SLURM_ARRAY_TASK_ID "30"		1000    1000     416  "Bayes-SA" 1 0 10 0
# python3 complexity_revisions.py $SLURM_ARRAY_TASK_ID "80"		1000    1000     416  "Bayes-SA" 0 0 10 0
# python3 complexity_revisions.py $SLURM_ARRAY_TASK_ID "30"		1000    1000     416  "Bayes-SA" 0 0 10 0


#													n_trials   	n_states  n_params  prob
# python3 probability_space.py $SLURM_ARRAY_TASK_ID 	500 		5000      100 	    90

# python3 complexity.py $SLURM_ARRAY_TASK_ID "rich"		500 		5000 		100
# python3 complexity.py $SLURM_ARRAY_TASK_ID "lean"		500 		5000 		100	

# 							job            	env   		n_trials	n_states	split	variant
# python3 complexity.py $SLURM_ARRAY_TASK_ID "rich"		500 		5000 		100		
# python3 complexity.py $SLURM_ARRAY_TASK_ID "lean"		500 		5000 		100		
# python3 complexity.py $SLURM_ARRAY_TASK_ID "rich_90_80"	500 		5000 		100		
# python3 complexity.py $SLURM_ARRAY_TASK_ID "rich_90_70"	500 		5000 		100		
# python3 complexity.py $SLURM_ARRAY_TASK_ID "lean_20_10"	500 		5000 		100		
# python3 complexity.py $SLURM_ARRAY_TASK_ID "lean_30_10"	500 		5000 		100		

# # variants
# python3 complexity.py $SLURM_ARRAY_TASK_ID "rich"		500 		5000 		100		"lrate"
# python3 complexity.py $SLURM_ARRAY_TASK_ID "rich_90_80"	500 		5000 		100		"lrate"
# python3 complexity.py $SLURM_ARRAY_TASK_ID "rich_90_70"	500 		5000 		100		"lrate"
# python3 complexity.py $SLURM_ARRAY_TASK_ID "lean"		500 		5000 		100		"lrate"
# python3 complexity.py $SLURM_ARRAY_TASK_ID "lean_20_10"	500 		5000 		100		"lrate"
# python3 complexity.py $SLURM_ARRAY_TASK_ID "lean_30_10"	500 		5000 		100		"lrate"	
