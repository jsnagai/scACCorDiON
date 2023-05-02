import numpy as np
import scipy.linalg as slg
import networkx as nx
from sklearn import preprocessing
#from numba import jit



def directed_res(G,weight='weight'):
    I=np.matrix(np.eye(len(G.nodes())))
    A = nx.adjacency_matrix(G,weight=weight).todense()
    D =  list(dict(G.out_degree()).values())*np.eye(A.shape[1])
    L = D-A
    Pi =  np.eye(A.shape[1]) - (1/A.shape[1])*np.ones([A.shape[1],A.shape[1]])
    # Here we can use hermitian, once Pi will be hermitian A = A^{T}
    eigPi = np.linalg.eigh(Pi) 
    Qi = eigPi[1][:,1:Pi.shape[1]+1]
    Qi = Qi.T
    rL = Qi @ L @ Qi.T
    Sig = slg.solve_continuous_lyapunov(a=rL,q=np.eye(A.shape[1]-1))
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
    assert(isinstance(G, nx.classes.digraph.DiGraph) or isinstance(G, nx.classes.multigraph.MultiDiGraph()), "Not a Digraph")
    X = directed_res(G,weight)[0]
    d = np.zeros(X.shape)
    for i in range(d.shape[0]):
        for j in range(i,d.shape[0]):
            if i != j:
               d[i,j]=X[i,i] + X[j,j] - 2*X[i,j]
               d[i,j]=np.sqrt(d[i,j])
               d[j,i]=d[i,j]
    return(np.real(np.round(d,4)))