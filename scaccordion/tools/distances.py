import numpy as np
import scipy.linalg as slg
import networkx as nx
from sklearn import preprocessing
from scipy.sparse import spdiags, diags
from scipy.sparse.linalg import eigs,eigsh  
#from numba import jit
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import eigsh
from scipy.linalg import  norm
from sklearn import manifold


def directed_res(G,weight='weight'):
    I=np.matrix(np.eye(len(G.nodes())))
    A = nx.adjacency_matrix(G,weight=weight).todense()
    n = A.shape[0]
    D = list(dict(G.out_degree()).values())*np.eye(n)
    L = D-A
    Pi =  np.eye(n) - (1/n)*np.ones([n,n])
    # Here we can use hermitian, once Pi will be hermitian A = A^{T}
    eigPi = np.linalg.eigh(Pi) 
    eigRes=eigPi[1][:,np.argsort(eigPi[0])]
    Qi = eigRes[:,1:Pi.shape[1]].T
    rL = Qi @ L @ Qi.T
    Sig = slg.solve_continuous_lyapunov(a=rL,q=np.eye(A.shape[1]-1)) #Matlab version needs to be -I instead I
    X = 2*Qi.T @ Sig @ Qi
    return(X,rL,Qi,Pi,L,A)



def ctd_dist(G,weight='weight'):
    """
        Compute the Commute Time Distance   
    

    Parameters
    ----------
    G : NetworkX DiGraph
       An undirected graph
    weight : string, function, or None
        If this is a string, then edge weights will be accessed via the
        edge attribute with this key (that is, the weight of the edge
        joining `u` to `v` will be ``G.edges[u, v][weight]``). If no
        such edge attribute exists, the weight of the edge is assumed to
        be one.
        If this is a function, the weight of an edge is the value
        returned by the function. The function must accept exactly three
        positional arguments: the two endpoints of an edge and the
        dictionary of edge attributes for that edge. The function must
        return a number.
        If this is None, every edge has weight/distance/cost 1.
        Weights stored as floating point values can lead to small round-off
        errors in distances. Use integer weights to avoid this.
        Weights should be positive, since they are distances.
    Returns
    -------
    matrix : node to node distance matrix


    Notes
    -----
    This algorithm was proposed in [1]_ and [2]_.
    https://github.com/zboyd2/hitting_probabilities_metric/tree/11c046b89b73944223106584abd577db49fcd2a5
    References
    ----------
    .. [1] Young, G. F., Scardovi, L., & Leonard, N. E.,
       "A new notion of effective resistance for directed graphs—Part I: Definition and properties"
       IEEE Transactions on Automatic Control, 2016
       https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7276998

    .. [2] Young, G. F., Scardovi, L., & Leonard, N. E.,
       "A new notion of effective resistance for directed graphs—Part II: Computing Resistances"
       IEEE Transactions on Automatic Control, 2016
       https://ieeexplore.ieee.org/stamp/stamp.jsp?arnumber=7276998
    """
    X = directed_res(G,weight)[0]
    d = np.zeros(X.shape)
    for i in range(d.shape[0]):
        for j in range(i,d.shape[0]):
            if i != j:
               d[i,j]=X[i,i] + X[j,j] - 2*X[i,j]
               d[i,j]=np.sqrt(d[i,j])
               d[j,i]=d[i,j]
    return(np.real(np.round(d,4)))



def get_Q_matrix(M):
    # taken from https://github.com/zboyd2/hitting_probabilities_metric/blob/master/HittingTimes_L3.m
    N = M.shape[0]
    e1 = np.zeros((1, N))
    e1[0][0] = 1
    A1inv = np.eye(N) - M
    A1inv[0, :] = e1
    A1inv = np.linalg.inv(A1inv)
    Q = np.zeros((N, N),dtype='double')
    Q[:, 0] = A1inv[:, 0].reshape(N,) / np.diagonal(A1inv)
    M = M @ A1inv
    detCj = ((1 + np.diagonal(M)) * (1 - M[0, 0])) + np.multiply(M[:, 0],M[0, :].transpose()).reshape(N,)
    CjInv = np.zeros((2, 2, N))
    CjInv[0, 0, :] = (1 - M[0, 0]) / detCj
    CjInv[0, 1, :] = M[:, 0].reshape(N,) / detCj # WAS WRONG INDEX HERE
    CjInv[1, 0, :] = -M[0, :].transpose().reshape(N,) / detCj
    CjInv[1, 1, :] = (1 + np.diagonal(M)) / detCj
    M1 = np.zeros((N, 2, N),dtype='double')
    M1[:, 0, :] = A1inv
    M1[:, 1, :] = np.tile(-A1inv[:, 0],( 1, N)).reshape(N, N)
    M2 = np.zeros((2, N, N),dtype='double')
    M2[0, :, :] = M.transpose()
    M2[1, :, :] = np.tile(M[0, :],(N, 1)).transpose()
    for j in range(1, N):
        # assert np.all(
        #    np.concatenate([A1inv[:, j].reshape((N, 1)), - A1inv[:, 0].reshape((N, 1))], axis=1) == M1[:, :, j])
        Ac = A1inv[:, j].reshape(N,) - np.matmul(np.matmul(M1[:, :, j], CjInv[:, :, j]), M2[:, j, j])
        Ad = np.diagonal(A1inv) - sum(
            M1[:, :, j].transpose() * np.matmul(CjInv[:, :, j], M2[:, :, j]))
        Q[:, j] = Ac / Ad
    np.fill_diagonal(Q, 0)
    return Q  # - np.diagonal(np.diagonal(Q))

def get_dhp(P, beta=0.5):
    n = P.shape[0]
    # Compute hitting probabilities
    Q = np.round(get_Q_matrix(P),5)
    # Find the invariant measure
    v = eigs(P.T.astype('f'), 
              k=1, 
              sigma=1+1e-6, 
              which='LM',
              tol=1e-3,maxiter=100000,v0=np.ones(n)/n)[1].ravel()
    v1 = np.abs(v)/norm(v,ord=1)
    #v = v-np.min(v) + 1e-10
    # Construct the symmetric adjacency matrix M:
    if beta == 0.5:
        Aht = diags(v1**0.5, 0,(n, n)) @ Q @ diags(v1**-0.5, 0,(n, n))
    elif beta == 1:
        Aht = diags(v1, 0,(n, n)) @ Q 
    else:
        raise ValueError(f"Unsupported beta value: {beta}")
    Aht = (Aht + Aht.T) / 2
    #np.fill_diagonal(Aht, 0)   
    # Compute the hitting probability distance
    dhp = -np.log10(Aht,where=Aht != 0)
    np.fill_diagonal(dhp,0)
    return(np.round(dhp,5)) 