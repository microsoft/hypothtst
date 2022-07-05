import numpy as np
from collections import Counter
from scipy.stats import chi2


def get_pval_sim(ni, ps=None, t_stat_fn=max):
	k = len(ni)
	n = sum(ni)
	if ps is None:
		ps = np.ones(k)/k
	t_stat = t_stat_fn(ni)
	return get_pval_sim2(t_stat, n, k, t_stat_fn, 1000, ps)


def get_pval_sim2(tst_stat=4, n=10, k=5, t_stat_fn=max,
				  n_sim=1000, ps=None):
	"""
	args:
		tst_stat: The test statistic calculated from the data.
		n: Number of packet drops
		k: Number of participants in the call
	"""
	if ps is None:
		ps = np.ones(k)/k
	p_val = 0
	for _ in range(n_sim):
		ys = np.random.choice(k, p=ps, size=n)
		ns = Counter(ys).values()
		ns = list(ns)
		sim_tst_stat = t_stat_fn(ns)
		if sim_tst_stat > tst_stat:
			p_val += 1/n_sim
	return p_val


def get_p_val_chisq(ni, ps=None):
	t_stat = t_stat_fn_1(ni, ps)
	k = len(ni)
	return chi2.sf(t_stat, df=k-1)


def t_stat_fn_1(ni, ps=None):
	"""
	For large n, this test stat follows the chi squared distribution.
	"""
	k = len(ni)
	n = sum(ni)
	if ps is None:
		ps = np.ones(k)/k
	t_stat = sum((ni - n*ps)**2/(n*ps))
	return t_stat
