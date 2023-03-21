import numpy as np
import re

def calc_probs(env):
	"""
	env - takes form of optprob_step_noptions
		e.g. 90_10_3
		probs = [90, 80, 80]

		e.g. 60_20_4
		probs = [60, 40, 40, 40]
	"""
	opt, step, noptions = [eval(x) for x in re.split("_",env)]

	probs = np.zeros(noptions) + opt/100
	probs[1:] = probs[1:] - step/100
	return probs

def get_probs(env):
	""" return reward contigencies of specified
	envrionemtns for each option.
	first option is always option with highest 
	reward feedback
	"""

	# hardcoded probs
	if env[0] == "r" or env[0] == "l" or env[0] == "m":

		# base environments
		if env == "rich":
			probs = np.array([.8, .7])
		elif env == "lean":
			probs = np.array([.3, .2])

		# varying complexity
		elif env == "rich_3":
			probs = np.array([.8, .7, .7])
		elif env == "rich_4":
			probs = np.array([.8, .7, .7, .7])
		elif env == "rich_5":
			probs = np.array([.8, .7, .7, .7, .7])
		elif env == "rich_6":
			probs = np.array([.8, .7, .7, .7, .7, .7])
		elif env == "rich_7":
			probs = np.array([.8, .7, .7, .7, .7, .7, .7])
		elif env == "rich_8":
			probs = np.array([.8, .7, .7, .7, .7, .7, .7, .7])
		elif env == "rich_9":
			probs = np.array([.8, .7, .7, .7, .7, .7, .7, .7, .7])

		elif env == "lean_3":
			probs = np.array([.3, .2, .2])
		elif env == "lean_4":
			probs = np.array([.3, .2, .2, .2])
		elif env == "lean_5":
			probs = np.array([.3, .2, .2, .2, .2])
		elif env == "lean_6":
			probs = np.array([.3, .2, .2, .2, .2, .2])
		elif env == "lean_7":
			probs = np.array([.3, .2, .2, .2, .2, .2, .2])
		elif env == "lean_8":
			probs = np.array([.3, .2, .2, .2, .2, .2, .2, .2])
		elif env == "lean_9":
			probs = np.array([.3, .2, .2, .2, .2, .2, .2, .2, .2])

		# varying discrimination
		elif env == "rich_hard":
			probs = np.array([.8, .75])
		elif env == "lean_hard":
			probs = np.array([.25, .2])
		elif env == "rich_easy":
			probs = np.array([.8, .6])
		elif env == "lean_easy":
			probs = np.array([.4, .2])

		# multi-option
		# rich
		elif env == "rich_90_80_70":
			probs = np.array([.9, .8, .7])
		elif env == "rich_90_70_50":
			probs = np.array([.9, .7, .5])
		elif env == "rich_80_70_60":
			probs = np.array([.8, .7, .6])
		elif env == "rich_80_60_40":
			probs = np.array([.8, .6, .4])
		elif env == "rich_70_60_50":
			probs = np.array([.7, .6, .5])
		elif env == "rich_70_50_30":
			probs = np.array([.7, .5, .3])
		elif env == "rich_80_70_30":
			probs = np.array([.8, .7, .3])

		# lean
		elif env == "lean_30_20_10":
			probs = np.array([.3, .2, .1])
		elif env == "lean_30_20_00":
			probs = np.array([.3, .2, .0])
		elif env == "lean_40_30_20":
			probs = np.array([.4, .3, .2])

		
		# varying complexity, extreme range
		elif env == "rich_90_80":
			probs = np.array([.9, .8])
		elif env == "rich_90_80_3":
			probs = np.array([.9, .8, .8])
		elif env == "rich_90_80_4":
			probs = np.array([.9, .8, .8, .8])
		elif env == "rich_90_80_5":
			probs = np.array([.9, .8, .8, .8, .8])
		elif env == "rich_90_80_6":
			probs = np.array([.9, .8, .8, .8, .8, .8])
		elif env == "rich_90_80_7":
			probs = np.array([.9, .8, .8, .8, .8, .8, .8])
		elif env == "rich_90_80_8":
			probs = np.array([.9, .8, .8, .8, .8, .8, .8, .8])
		elif env == "rich_90_80_9":
			probs = np.array([.9, .8, .8, .8, .8, .8, .8, .8, .8])

		# varying complexity,  higher discrim
		elif env == "rich_90_70":
			probs = np.array([.9, .7])
		elif env == "rich_90_70_3":
			probs = np.array([.9, .7, .7])
		elif env == "rich_90_70_4":
			probs = np.array([.9, .7, .7, .7])
		elif env == "rich_90_70_5":
			probs = np.array([.9, .7, .7, .7, .7])
		elif env == "rich_90_70_6":
			probs = np.array([.9, .7, .7, .7, .7, .7])
		elif env == "rich_90_70_7":
			probs = np.array([.9, .7, .7, .7, .7, .7, .7])
		elif env == "rich_90_70_8":
			probs = np.array([.9, .7, .7, .7, .7, .7, .7, .7])
		elif env == "rich_90_70_9":
			probs = np.array([.9, .7, .7, .7, .7, .7, .7, .7, .7])

		elif env == "richmix":	# duplicated from rich for integration ease
			probs = np.array([.8, .7])
		elif env == "richmix3":
			probs = np.array([.8, .7, .6])
		elif env == "richmix4":
			probs = np.array([.8, .7, .6, .7])
		elif env == "richmix5":
			probs = np.array([.8, .7, .6, .7, .6])
		elif env == "richmix6":
			probs = np.array([.8, .7, .6, .7, .6, .7])
		elif env == "richmix7":
			probs = np.array([.8, .7, .6, .7, .6, .7, .6])
		elif env == "richmix8":
			probs = np.array([.8, .7, .6, .7, .6, .7, .6, .7])
		elif env == "richmix9":
			probs = np.array([.8, .7, .6, .7, .6, .7, .6, .7, .6])

		# varying complexity, extreme range
		elif env == "lean_20_10":
			probs = np.array([.2, .1])
		elif env == "lean_20_10_3":
			probs = np.array([.2, .1, .1])
		elif env == "lean_20_10_4":
			probs = np.array([.2, .1, .1, .1])
		elif env == "lean_20_10_5":
			probs = np.array([.2, .1, .1, .1, .1])
		elif env == "lean_20_10_6":
			probs = np.array([.2, .1, .1, .1, .1, .1])
		elif env == "lean_20_10_7":
			probs = np.array([.2, .1, .1, .1, .1, .1, .1])
		elif env == "lean_20_10_8":
			probs = np.array([.2, .1, .1, .1, .1, .1, .1, .1])
		elif env == "lean_20_10_9":
			probs = np.array([.2, .1, .1, .1, .1, .1, .1, .1, .1])

		# varying complexity, higher discrim
		elif env == "lean_30_10":
			probs = np.array([.3, .1])
		elif env == "lean_30_10_3":
			probs = np.array([.3, .1, .1])
		elif env == "lean_30_10_4":
			probs = np.array([.3, .1, .1, .1])
		elif env == "lean_30_10_5":
			probs = np.array([.3, .1, .1, .1, .1])
		elif env == "lean_30_10_6":
			probs = np.array([.3, .1, .1, .1, .1, .1])
		elif env == "lean_30_10_7":
			probs = np.array([.3, .1, .1, .1, .1, .1, .1])
		elif env == "lean_30_10_8":
			probs = np.array([.3, .1, .1, .1, .1, .1, .1, .1])
		elif env == "lean_30_10_9":
			probs = np.array([.3, .1, .1, .1, .1, .1, .1, .1, .1])

		elif env == "leanmix":	# duplicated from lean for integration ease
			probs = np.array([.3, .2])
		elif env == "leanmix3":
			probs = np.array([.3, .2, .1])
		elif env == "leanmix4":
			probs = np.array([.3, .2, .1, .2])
		elif env == "leanmix5":
			probs = np.array([.3, .2, .1, .2, .1])
		elif env == "leanmix6":
			probs = np.array([.3, .2, .1, .2, .1, .2])
		elif env == "leanmix7":
			probs = np.array([.3, .2, .1, .2, .1, .2, .1])
		elif env == "leanmix8":
			probs = np.array([.3, .2, .1, .2, .1, .2, .1, .2])
		elif env == "leanmix9":
			probs = np.array([.3, .2, .1, .2, .1, .2, .1, .2, .1])

		elif env == "mix":	# duplicated from rich for integration ease
			probs = np.array([.8, .7])
		elif env == "mix3":
			probs = np.array([.8, .7, .6])
		elif env == "mix4":
			probs = np.array([.8, .7, .6, .5])
		elif env == "mix5":
			probs = np.array([.8, .7, .6, .5, .4])
		elif env == "mix6":
			probs = np.array([.8, .7, .6, .5, .4, .3])
		elif env == "mix7":
			probs = np.array([.8, .7, .6, .5, .4, .3, .2])
		elif env == "mix8":
			probs = np.array([.8, .7, .6, .5, .4, .3, .2, .1])
		elif env == "mix9":
			probs = np.array([.8, .7, .6, .5, .4, .3, .2, .1, .0])


		else:
			err = 'Invalid value given for arg env. %s given' %env
			raise Exception(err)

	# probs are explivitly given
	elif type(env) != str:
		probs = env

	# calculate probs on the fly
	else:
		probs = calc_probs(env)

	return probs