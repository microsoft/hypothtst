from scipy.stats import binom_test, poisson, binom
from scipy.special import comb
import numpy as np
from scipy.optimize import root_scalar

def binom_tst_beta(p_null=0.5,p_alt=0.6,n=10,alpha_hat=0.05):
    if n==0:
        return 1.0
    x_a = binom.isf(alpha_hat,n,p_null)
    return binom.cdf(x_a,n,p_alt)


def binom_tst_alpha(hat_alpha=0.5,p=0.4,n=10):
    return binom.sf(binom.isf(hat_alpha,p=p,n=n),p=p,n=n)


def binom_tst_beta_sim(p_null=0.5,p_alt=0.6,n=10,alpha_hat=0.05,n_sim=1000):
    #Generate from the alternate.
    rvs = binom.rvs(n,p_alt,size=n_sim)
    #Check against the null.
    p_vals = np.array([binom_test(i,n,p_null,alternative='greater') \
                for i in rvs])
    return sum(p_vals>alpha_hat)/len(rvs)

## https://github.com/ryu577/scipy/pull/1/commits/9630185a7dac681497cb4a83958eb335f5938aca
def binom_test_v2(x, n=None, p=0.5, alternative='two-sided'):
    """
    Perform a test that the probability of success is p.
    This is an exact, two-sided test of the null hypothesis
    that the probability of success in a Bernoulli experiment
    is `p`.
    Parameters
    ----------
    x : int or array_like
        The number of successes, or if x has length 2, it is the
        number of successes and the number of failures.
    n : int
        The number of trials.  This is ignored if x gives both the
        number of successes and failures.
    p : float, optional
        The hypothesized probability of success.  ``0 <= p <= 1``. The
        default value is ``p = 0.5``.
    alternative : {'two-sided', 'greater', 'less'}, optional
        Indicates the alternative hypothesis. The default value is
        'two-sided'.
    Returns
    -------
    p-value : float
        The p-value of the hypothesis test.
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Binomial_test
    Examples
    --------
    >>> from scipy import stats
    A car manufacturer claims that no more than 10% of their cars are unsafe.
    15 cars are inspected for safety, 3 were found to be unsafe. Test the
    manufacturer's claim:
    >>> stats.binom_test(3, n=15, p=0.1, alternative='greater')
    0.18406106910639114
    The null hypothesis cannot be rejected at the 5% level of significance
    because the returned p-value is greater than the critical value of 5%.
    """
    #x = atleast_1d(x).astype(np.int_)
    # if len(x) == 2:
    #     n = x[1] + x[0]
    #     x = x[0]
    # elif len(x) == 1:
    #     x = x[0]
    #     if n is None or n < x:
    #         raise ValueError("n must be >= x")
    #     n = np.int_(n)
    # else:
    #     raise ValueError("Incorrect length for x.")
    n = np.int_(n)
    if (p > 1.0) or (p < 0.0):
        raise ValueError("p must be in range [0,1]")

    if alternative not in ('two-sided', 'less', 'greater'):
        raise ValueError("alternative not recognized\n"
                         "should be 'two-sided', 'less' or 'greater'")

    if alternative == 'less':
        pval = binom.cdf(x, n, p)
        return pval

    if alternative == 'greater':
        pval = binom.sf(x-1, n, p)
        return pval

    # if alternative was neither 'less' nor 'greater', then it's 'two-sided'
    d = binom.pmf(x, n, p)
    rerr = 1 + 1e-7
    a_fn = lambda x1:binom.pmf(x1,n,p)
    if x == p * n:
        # special case as shortcut, would also be handled by `else` below
        pval = 1.
    elif x < p * n:
        #i = np.arange(np.ceil(p * n), n+1)
        #y = np.sum(binom.pmf(i, n, p) <= d*rerr, axis=0)        
        ix = binary_search4(lambda x1:-binom.pmf(x1,n,p),-d*rerr,np.ceil(p * n),n)
        y = n-ix + int(d*rerr == binom.pmf(ix, n, p))        
        pval = (binom.cdf(x, n, p) +
                binom.sf(n - y, n, p))
    else:
        #i = np.arange(np.floor(p*n) + 1)
        #y = np.sum(binom.pmf(i, n, p) <= d*rerr, axis=0)
        ix = binary_search4(a_fn,d*rerr,0,np.floor(p*n))
        y = ix + 1
        pval = (binom.cdf(y-1, n, p) +
                binom.sf(x-1, n, p))

    return min(1.0, pval)


def _binary_search_for_binom_tst2(d, lo, hi):
    def f(k, d):
        k_low = np.minimum(np.floor(k), n-1)
        a, b = binom.pmf((k_low, k_low+1), n, p)
        return (0 if (a <= d <= b) or (a >= d >= b)
                else a + (k - k_low) * (b - a) - d)  # interpolate
    return root_scalar(f, bracket=(lo, hi), args=(d,)).root

def _binary_search_for_binom_tst3(d, lo, hi):
    def f(i, d):
        i_low = np.minimum(np.floor(i), n-1)
        a, b = binom.pmf((i_low, i_low+1), n, p)
        c = a + (i - i_low) * (b - a)
        return c - d
    return root_scalar(f, bracket=(lo, hi), args=(d,)).root

def binary_search(a, d, lo, hi, asc_order=False):
    while lo < hi:
        mid = (lo+hi)//2
        midval = a(mid)
        if midval < d:
            if asc_order:
                lo = mid+1
            else:
                hi = mid-1
        elif midval > d:
            if asc_order:
                hi = mid-1
            else:
                lo = mid+1
        else:
            return mid
    if a(lo)<=d:
        print("condition 1")
        return lo
    else:
        print("condition 2")
        return lo-(asc_order-0.5)*2

def binary_search4(a, d, lo, hi):
    while lo < hi:
        mid = (lo+hi)//2
        midval = a(mid)
        if midval < d:
            lo = mid+1
        elif midval > d:
            hi = mid-1
        else:
            return mid
    if a(lo)<=d:
        print("condition 1")
        return lo
    else:
        print("condition 2")
        return lo-1


def tst_alternate_search():
    import time
    n=50000000; p=0.4
    d = (binom.pmf(int(n/2-n/4),n,p)+binom.pmf(int(n/2-n/4)+1,n,p))/2
    tic = time.perf_counter()
    _binary_search_for_binom_tst2(d,0,np.floor(n*p)+1)
    toc = time.perf_counter()
    print(f"Fn took {toc - tic:0.4f} seconds")

    a_fn = lambda x1:binom.pmf(x1, n, p)
    tic = time.perf_counter()
    binary_search(a_fn, d, 0, np.floor(n*p)+1, True)
    toc = time.perf_counter()
    print(f"Fn took {toc - tic:0.4f} seconds")


def tst_binom_v2_low():
    p_val1 = binom_test(9,21,0.48)
    p_val1 = 0.6689672431938848
    p_val2 = binom_test_v2(9,21,0.48)
    print(p_val1==p_val2)
    p_val3 =  binom_test(10079999,21000000,0.48)
    p_val3 = 0.979042561004596
    p_val4 =  binom_test_v2(10079999,21000000,0.48)
    print(p_val3==p_val4)
    p_val5 =  binom_test(10079990,21000000,0.48)
    p_val5 = 0.9785298857599378
    p_val6 =  binom_test_v2(10079990,21000000,0.48)
    print(p_val5==p_val6)
    p_val7 = binom_test(4,21,0.48)
    p_val7 = 0.008139563452105921
    p_val8 = binom_test_v2(4,21,0.48)
    print(p_val7==p_val8)
    return p_val1 == p_val2


def tst_binom_v2_hi():
    p_val1 = binom_test(11,21,0.48)
    p_val1 = 0.8278629664608201
    p_val2 = binom_test_v2(11,21,0.48)
    print(p_val1==p_val2)
    p_val3 =  binom_test(10080009,21000000,0.48)
    p_val3 = 0.9786038762958954
    p_val4 =  binom_test_v2(10080009,21000000,0.48)
    print(p_val3==p_val4)
    p_val5 =  binom_test(10080017,21000000,0.48)
    p_val5 = 0.9778567637538729
    p_val6 =  binom_test_v2(10080017,21000000,0.48)
    print(p_val5==p_val6)
    p_val7 = binom_test(7,21,0.48)
    p_val7 = 0.19667729017182273
    p_val8 = binom_test_v2(7,21,0.48)
    print(p_val7==p_val8)
    return p_val1 == p_val2


def test_binomtest():
    """
    Some tests for binomtest. Should print all True when run.
    """
    pval1 = binomtest(9, 21, 0.48).pvalue
    print(pval1 == 0.6689672431938848)
    p_val3 = binomtest(10079999, 21000000, 0.48).pvalue
    print(p_val3 == 0.979042561004596)
    p_val5 = binomtest(10079990, 21000000, 0.48).pvalue
    print(p_val5 == 0.9785298857599378)
    p_val7 = binomtest(4, 21, 0.48).pvalue
    print(p_val7 == 0.008139563452105921)
    p_val9 = binomtest(11, 21, 0.48).pvalue
    print(p_val9 == 0.8278629664608201)
    p_val11 = binomtest(10080009, 21000000, 0.48).pvalue
    print(p_val11 == 0.9786038762958954)
    p_val13 = binomtest(10080017, 21000000, 0.48).pvalue
    print(p_val13 == 0.9778567637538729)
    p_val15 = binomtest(7, 21, 0.48).pvalue
    print(p_val15 == 0.19667729017182273)

def tst_edge():
    pv = binom_test_v2(484, 967, 0.5)
    print(pv == 0.999999999998212)
    pv = binom_test_v2(3, 47, 3/47)
    print(pv==0.9999999999999998)
    pv = binom_test_v2(13, 46, 13/46)
    print(pv==0.9999999999999987)
    pv = binom_test_v2(15, 44, 15/44)
    print(pv==0.9999999999999989)
    pv = binom_test_v2(7, 13, 0.5)
    print(pv==0.9999999999999999)
    pv = binom_test_v2(6, 11, 0.5)
    print(pv==0.9999999999999997)
