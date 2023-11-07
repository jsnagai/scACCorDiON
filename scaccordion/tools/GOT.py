import numpy as np
import numpy.linalg as lg
import scipy.linalg as slg


def wass_dist_(A, B):
    """
        Compute the wasserstein distance using undirected graph laplacians   
    Parameters
    ----------
    A,B : numpy matrix
    
    Returns
    -------
    dist : Wasserstein distance


    Notes
    -----
    This implementation is provided by _[1] .
    https://github.com/Hermina/GOT/blob/master/GOT_structural_example.ipynb
    References
    ----------
    .. [1] Petric Maretic, Hermina, et. al.
       "GOT: an optimal transport framework for graph comparison"
        Advances in Neural Information Processing Systems, 32, 2019
       https://papers.neurips.cc/paper_files/paper/2019/file/fdd5b16fc8134339089ef25b3cf0e588-Paper.pdf
    """
    n = len(A)
    l1_tilde = A + np.ones([n,n])/n #adding 1 to zero eigenvalue; does not change results, but is faster and more stable
    l2_tilde = B + np.ones([n,n])/n #adding 1 to zero eigenvalue; does not change results, but is faster and more stable
    s1_tilde = lg.inv(l1_tilde)
    s2_tilde = lg.inv(l2_tilde)
    Root_1= slg.sqrtm(s1_tilde)
    Root_2= slg.sqrtm(s2_tilde)
    return np.trace(s1_tilde) + np.trace(s2_tilde) - 2*np.trace(slg.sqrtm(Root_1 @ s2_tilde @ Root_1)) 