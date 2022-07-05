import numpy as np


def rejectn_rate(p_vals, alpha_hats=np.arange(0, 1.00001, 0.00001)):
    rejectn = np.zeros(len(alpha_hats))
    for p_val in p_vals:
        rejectn += (p_val <= alpha_hats) / len(p_vals)
    return rejectn


class AlphaBetaSim(object):
    def __init__(self, alpha_hats=None):
        if alpha_hats is not None:
            self.alpha_hats = alpha_hats
        else:
            self.alpha_hats = np.concatenate((np.arange(\
                0.000000000001,0.0099,0.0000001),
                np.arange(0.01,1.00,0.001),
                np.arange(0.991,1.00,0.001)),axis=0)

    def alpha_beta_tracer(self,null,alter,tst_null,tst_alt=None,n_sim=10000,debug=True):
        if tst_alt is None:
            tst_alt = tst_null
        self.alphas = self.rejection_rate(null,null,tst_null,n_sim)
        self.betas = 1-self.rejection_rate(null,alter,tst_alt,n_sim,debug=debug)
        return self.alphas, self.betas

    def alpha_beta_tracer2(self,null1,null2,alter,tst,n_sim=10000,debug=True):
        self.alphas = self.rejection_rate(null1,null2,tst,n_sim)
        self.betas = 1-self.rejection_rate(null1,alter,tst,n_sim,debug=debug)
        return self.alphas, self.betas

    def rejection_rate(self,dist1,dist2,tst,n_sim=10000,debug=False):
        """
        At what rate is the null hypothesis that some property of two
        distributions is the same getting rejected?
        """
        rejectn_rate = np.zeros(len(self.alpha_hats))
        if debug:
            m1s = []; m2s = []
        # First generate from null and find alpha_hat and alpha.
        for _ in range(n_sim):
            m1 = dist1.rvs()
            m2 = dist2.rvs()
            if debug:
                m1s.append(m1)
                m2s.append(m2)
            p_val = tst(m1,m2)
            rejectn_rate += (p_val < self.alpha_hats)/n_sim
        if debug:
            self.m1s = np.array(m1s); self.m2s=np.array(m2s)
        return rejectn_rate

    def beta(self,alpha):
        ix=np.argmin((self.alphas-alpha)**2)
        return self.alphas[ix], self.betas[ix]


class AlphaBeta_1Sampl(object):
    def __init__(self, alpha_hats=None):
        if alpha_hats is not None:
            self.alpha_hats = alpha_hats
        else:
            self.alpha_hats = np.concatenate((np.arange(
                0.000000000001, 0.0099, 0.0000001),
                np.arange(0.01, 1.00, 0.001),
                np.arange(0.991, 1.00, 0.001)), axis=0)

    def alpha_beta_tracer(self, null, alter,
                          tst_null, tst_alt=None, n_sim=10000):
        if tst_alt is None:
            tst_alt = tst_null
        self.alphas = self.rejection_rate(null, tst_null, n_sim)
        self.betas = 1-self.rejection_rate(alter, tst_alt, n_sim)

    def rejection_rate(self,dist1,tst,n_sim=10000):
        """
        At what rate is the null hypothesis that some property of two
        distributions is the same getting rejected?
        """
        rejectn_rate = np.zeros(len(self.alpha_hats))
        # First generate from null and find alpha_hat and alpha.
        for _ in range(n_sim):
            m1 = dist1.rvs()
            try:
                p_val = tst(m1)
            except:
                print("Failed for array: " + str(m1))
            rejectn_rate += (p_val < self.alpha_hats)/n_sim
        return rejectn_rate

    def beta(self, alpha):
        # TODO: replace this with binary search.
        ix = np.argmin((self.alphas-alpha)**2)
        return self.alphas[ix], self.betas[ix]
