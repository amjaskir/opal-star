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

# Run on Carney cluster
#SBATCH --account=carney-frankmj-condo

# Request specified runtime:
#SBATCH --time=2:00:00

# Default resources are 1 core with 2.8GB of memory.
# Use more memory (4GB):
##SBATCH --mem=20G

# Go through the designed ntrials
##SBATCH --array=50,100,200,500
#SBATCH --array=50,100,250,500,1000
##SBATCH --array=20,30,40,50,60,70,80,90

# Specify a job name:
#SBATCH -J ranked_SACrit_BayesAnneal

# Specify an output file
# %j is a special variable that is replaced by the JobID when 
# job starts
#SBATCH -o ./output_grid/ranked_SACrit_BayesAnneal_%j.out
#SBATCH -e ./output_grid/ranked_SACrit_BayesAnneal_%j.err

#----- End of slurm commands ----
echo "Starting job $SLURM_ARRAY_TASK_ID"

# python3 cmp_complexity_ntrials.py  "80"				500 		5000 		$SLURM_ARRAY_TASK_ID  "nohebb"
python3 complexity_hist_revisions.py "80"				1000 		1000        $SLURM_ARRAY_TASK_ID    "SA"    1   0   10  0 "bmod"
python3 complexity_hist_revisions.py "30"				1000 		1000 		$SLURM_ARRAY_TASK_ID    "SA"    1   0   10  0 "bmod"
# python3 final_figs.py  500     5000    500       


# python3 probability_plot.py $SLURM_ARRAY_TASK_ID
# python3 complexity_hist.py $SLURM_ARRAY_TASK_ID	500 5000 500  

# python3 opt_parameters.py $SLURM_ARRAY_TASK_ID

# comparing complexity

#							   env_base			n_trials 	n_states 	pltn  version	
# python3 cmp_complexity_hist2.py "rich"				500 		5000 		40    "bmod"
# python3 cmp_complexity_hist2.py "rich"				500 		5000 		100   "bmod"
# python3 cmp_complexity_hist2.py "rich"				500 		5000 		250   "bmod"
# python3 cmp_complexity_hist2.py "rich"				500 		5000 		500   "bmod"

# python3 cmp_complexity_hist2.py "lean"				500 		5000 		40    "bmod"
# python3 cmp_complexity_hist2.py "lean"				500 		5000 		100   "bmod"
# python3 cmp_complexity_hist2.py "lean"				500 		5000 		250   "bmod"
# python3 cmp_complexity_hist2.py "lean"				500 		5000 		500   "bmod"





# general complex analysis

# python3 complexity_hist.py "lean"				500 		5000 		40  
# python3 complexity_hist.py "rich"				500 		5000 		40  
# python3 complexity_hist.py "lean"				500 		5000 		50 
# python3 complexity_hist.py "rich"				500 		5000 		50 

# python3 complexity_hist.py "lean"				1000 		5000 		500 
# python3 complexity_hist.py "lean"				1000 		5000 		250 
# python3 complexity_hist.py "lean"				1000 		5000 		100 
# python3 complexity_hist.py "lean"				1000 		5000 		40    


# python3 complexity_hist.py "rich_90_80"		500 		5000 		40    "bmod"
# python3 complexity_hist.py "rich_90_80"		500 		5000 		100   "bmod"
# python3 complexity_hist.py "rich_90_80"		500 		5000 		250   "bmod"
# python3 complexity_hist.py "rich_90_80"		500 		5000 		500   "bmod"

# python3 complexity_hist.py "rich_90_70"		500 		5000 		40    "bmod"
# python3 complexity_hist.py "rich_90_70"		500 		5000 		100   "bmod"
# python3 complexity_hist.py "rich_90_70"		500 		5000 		250   "bmod"
# python3 complexity_hist.py "rich_90_70"		500 		5000 		500   "bmod"

# python3 complexity_hist.py "lean"				500 		5000 		40    "lrate"
# python3 complexity_hist.py "lean"				500 		5000 		100   "lrate"
# python3 complexity_hist.py "lean"				500 		5000 		250   "lrate"
# python3 complexity_hist.py "lean"				500 		5000 		500   "lrate"

# python3 complexity_hist.py "lean_20_10"		500 		5000 		40    "bmod"
# python3 complexity_hist.py "lean_20_10"		500 		5000 		100   "bmod"
# python3 complexity_hist.py "lean_20_10"		500 		5000 		250   "bmod"
# python3 complexity_hist.py "lean_20_10"		500 		5000 		500   "bmod"

# python3 complexity_hist.py "lean_30_10"		500 		5000 		40    "bmod"
# python3 complexity_hist.py "lean_30_10"		500 		5000 		100   "bmod"
# python3 complexity_hist.py "lean_30_10"		500 		5000 		250   "bmod"
# python3 complexity_hist.py "lean_30_10"		500 		5000 		500   "bmod"



# Complexity
#							env_base	n_trials 	n_states 	pltn  			
# python3 complexity_hist.py "rich"		500 		5000 		250   
# python3 complexity_hist.py "rich_90_80"	500 	5000 		250	  
# python3 complexity_hist.py "rich_90_70"	500 	5000 		250	  
# python3 complexity_hist.py "lean"			500 	5000 		250   
# python3 complexity_hist.py "lean_20_10"	500 	5000 		250   
# python3 complexity_hist.py "lean_30_10"	500 	5000 		250   

#							env_base	n_trials 	n_states 	pltn  version			
# python3 complexity_hist.py "rich"			500 	5000 		500   "bmod"
# python3 complexity_hist.py "rich_90_80"	500 	5000 		500	  "bmod"
# python3 complexity_hist.py "rich_90_70"	500 	5000 		500	  "bmod"
# python3 complexity_hist.py "lean"			500 	5000 		500   "bmod"
# python3 complexity_hist.py "lean_20_10"	500 	5000 		500   "bmod"
# python3 complexity_hist.py "lean_30_10"	500 	5000 		500   "bmod"



# Standard OpAL, thresh 10
# # 					  env norm       k       reprocess
# python3 hist_grid.py "lean" 0 $SLURM_ARRAY_TASK_ID 1
# python3 hist_grid.py "rich" 0 $SLURM_ARRAY_TASK_ID 1
# python3 hist_grid.py "rich_90_80_70" 0 $SLURM_ARRAY_TASK_ID 1
# python3 hist_grid.py "lean_30_20_10" 0 $SLURM_ARRAY_TASK_ID 1

# python3 hist_grid.py "lean" 1 $SLURM_ARRAY_TASK_ID 1
# python3 hist_grid.py "rich" 1 $SLURM_ARRAY_TASK_ID 1
# python3 hist_grid.py "rich_90_80_70" 1 $SLURM_ARRAY_TASK_ID 1
# python3 hist_grid.py "lean_30_20_10" 1 $SLURM_ARRAY_TASK_ID 1


# Bayesian Critic
# 					  		env 	norm       k       reprocess
# python3 hist_grid_bayes.py "lean" 	0 $SLURM_ARRAY_TASK_ID 	1 	
# python3 hist_grid_bayes.py "rich" 	0 $SLURM_ARRAY_TASK_ID 	1 	
# python3 hist_grid_bayes.py "rich_90_80_70" 0 $SLURM_ARRAY_TASK_ID 1 
# python3 hist_grid_bayes.py "lean_30_20_10" 0 $SLURM_ARRAY_TASK_ID 1 

# python3 hist_grid_bayes.py "lean" 	1 $SLURM_ARRAY_TASK_ID 	1 	
# python3 hist_grid_bayes.py "rich" 	1 $SLURM_ARRAY_TASK_ID 	1 	
# python3 hist_grid_bayes.py "rich_90_80_70" 1 $SLURM_ARRAY_TASK_ID 1 
# python3 hist_grid_bayes.py "lean_30_20_10" 1 $SLURM_ARRAY_TASK_ID 1 

