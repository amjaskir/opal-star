##################################################################
# Alana Jaskir
# Brown University, Providence, RI, USA
# Laboratory for Neural Computation and Cognition
# Michael Frank
#
# Decription: Compare effects of DA modulation in bayesian opal model
#
# Name: opal/hist_grid_bayes.py
##################################################################

import pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import itertools
import scipy.stats as stats

# get my helper modules
import sys
import os
sys.path.insert(1, os.path.join(sys.path[0], '../../helpers'))
import params
import aucplotter

def main(env,norm,k,reprocess):
	"""
	Graphs data outputed by grid_search.py into histograms

	env  		environment specified by environments.py
	norm  		whether PE is normalized or not
	k 			multiplier for modulation
	reprocess 	whether to reprocess cached graph data
	"""

	# which k to use
	k = float(k)
	lmbdas = np.arange(100,201,100)

	for lmbda in lmbdas:
		path_base = "results/grid_bayes/%s/norm_%s/k_0.0/l_%.1f/mod_constant/" \
					%(env,str(norm),lmbda)
		path = "results/grid_bayes/%s/norm_%s/k_%s/l_%.1f/" \
					%(env,str(norm),k,lmbda)

		# check if cached data exists
		dump_it = path + "saved.pkle"
		if os.path.exists(dump_it) and not reprocess:
			auc_learn_modT,auc_learn_modF,auc_reward_modT,auc_reward_modF,\
			auc_diff_learn,auc_diff_reward = \
			pickle.load(open(dump_it,"rb"))

		else:
			# all param combinations
			param_comb = params.get_grid_bayes()

			auc_learn_modT = []
			auc_reward_modT = []
			auc_learn_modF = []
			auc_reward_modF = []
			auc_diff_learn = []
			auc_diff_reward = []

			# save parameters to color diff by AUC graphs
			color_beta = []
			color_alphaa = []
			titles = ["Beta","Alpha A"]
			saveas = ["diff_by_auc_beta","diff_by_auc_alphaa"]
			maps = ["plasma","plasma"]	# use coolwarm for boolean

			for par in param_comb:
				#print('params: %s' % str(par))
				#sys.stdout.flush()
				par2 = par[1:3]

				# handle parameter ish
				alpha_a, beta = par2
				color_alphaa.append(alpha_a)
				color_beta.append(beta)

				if os.path.exists(path + "mod_beta/params_" + str(par2) + ".pkle"):
					# get modulated data
					auc_learnT, auc_rewardT, learn_curve, reward_curve = \
					pickle.load(open(path + "mod_beta/params_" + str(par2) + ".pkle","rb"))
					# get unmodulated data 
					auc_learnF, auc_rewardF, learn_curve, reward_curve = \
					pickle.load(open(path_base + "params_" + str(par2) + ".pkle","rb"))

					# save
					auc_learn_modT.append(auc_learnT)
					auc_learn_modF.append(auc_learnF)
					auc_reward_modT.append(auc_rewardT)
					auc_reward_modF.append(auc_rewardF)
					auc_diff_learn.append(auc_learnT - auc_learnF)
					auc_diff_reward.append(auc_rewardT - auc_rewardF)

				else:
					tried = str(path + "mod_beta/params_" + str(par2) + ".pkle")
					print(tried)
					print("missing params", str(par2))
			colors = (color_beta,color_alphaa)

			# save to regreaph later
			pickle.dump((auc_learn_modT,auc_learn_modF,auc_reward_modT,auc_reward_modF,\
				auc_diff_learn,auc_diff_reward,colors), \
				open(dump_it, "wb"))

		# Graph everything
		what = "mod"
		aucplotter.graph_learn(auc_learn_modT,auc_learn_modF,path,what)
		aucplotter.graph_reward(auc_reward_modT,auc_reward_modF,path,what)
		aucplotter.graph_diff(auc_diff_learn,path,"Learning",what)
		aucplotter.graph_diff(auc_diff_reward,path,"Reward",what)
		aucplotter.graph_diff_by_AUC(np.array(auc_learn_modF),np.array(auc_diff_learn),\
			colors,titles,saveas,maps,\
			path,"Learning",what)
		aucplotter.graph_diff_by_AUC(np.array(auc_reward_modF),np.array(auc_diff_reward),\
			colors,titles,saveas,maps,\
			path,"Reward",what)

if __name__ == '__main__':
	main(sys.argv[1],bool(int(sys.argv[2])),int(sys.argv[3]),bool(int(sys.argv[4])))





