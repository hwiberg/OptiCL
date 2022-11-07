import pandas as pd
import itertools
import os 
import numpy as np

df = pd.read_csv('../sh/failed_initial_batch.csv')

n_iterations = 100

for idx, row in df.iterrows():
	index = row['index']
	alg = row['alg']
	bs = row['bs']
	viol_rule = row['viol_rule']
	code_version = '%s_bs_%d_group_%s' % (alg, bs, str(viol_rule))
	print("\nAggregating code version %d = %s" % (index, code_version))
	try:
		df_temp = pd.read_csv('solution_%s.csv' % (code_version))
	except:
		print("--- still not done ---")
		os.system('mv solution_%s.csv patch_seeds/' % (code_version))
		sol_list =  []
		fail_list = []
		for seed in range(n_iterations):
			try: 
				x = pd.read_csv('patch_seeds/solution_%s_seed%d.csv' % (code_version, seed))
				sol_list.append(x)
			except:
				fail_list.append(seed)
		print("Failed jobs: %s" % fail_list)
		if len(fail_list) == 0:
			sol_full = pd.concat(sol_list, axis=0)
			sol_full.to_csv('solution_%s.csv' % (code_version), index = False)
		else: 
			print("index missing jobs - did not save")
			# sol_full = pd.concat(sol_list, axis=0)
			# sol_full.to_csv('solution_%s_partial.csv' % (code_version), index = False)