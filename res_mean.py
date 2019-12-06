import pandas as pd
import numpy as np
dataset = "BlogCatalog"# "Flickr"
dataset_ex_str = "1"
# for nin in ["0.0","1e-06","0.0001","0.01","1.0"]:
for nout in ["1","2","3"]:
	# extra_str = "0.6do0.1ep200lbd"+lbd+"nout2alp"+alp+"normy"
	extra_str = "0.6do0.1ep200lbd0.0001"
	if nout != "1":
		extra_str += "nout" + nout
	extra_str += "alp0.0001normy"
	# "nout"2 + "alp0.0001normy"
	fn = "./new_results/"+dataset+dataset_ex_str+"/"+extra_str+".csv"
	# print(alp, lbd)
	df = pd.read_csv(fn, header=None)
	# print(df.values.shape[0])
	mean = np.mean(df.values[:10,:],axis=0)
	print(np.round(mean,3))