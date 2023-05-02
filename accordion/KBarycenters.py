import pandas as pd
import numpy as np
import random

class KBarycenters():
    def __init__(self, k = 2, iter=300, seed=42):

def kbary1(data,nclusters,iters=300,seed=42):
    # Getting Initial Centroids
    random.seed(42)
    nsample = data.shape[0]
    cvec = data.index[random.sample(range(nsample), nclusters)].tolist()
    ncvec = cvec.copy()
    solutions = -1*np.ones(nsample) ## Set solution vector
    for i in range(nclusters):
        solutions[np.where(cvec[i]==data.index)[0]]=i ## Init Centroids in sol
    tmpsolutions = solutions.copy() ## Initi Aux in sol
    pbar = tqdm()
    visited = set()
    visited.add(tuple(sorted(cvec)))
    while iters != 0: # Criteria: Centroid were already used or the points are the same
        solutions=tmpsolutions.copy()
        for sample in range(nsample): # Assign the closest neighbors
             if solutions[sample]==-1:
                    solutions[sample]=np.argmin(data.loc[data.index[sample],cvec])
        tmpsolutions= -1*np.ones(nsample)
        for i in range(nclusters): # Update centroid/barycenter 
            #print(i)
            sel = data.index[solutions==i].tolist().copy()
            if np.sum(solutions==i) > 1: 
                if cvec[i] in sel:
                    sel.remove(cvec[i])
                bary = (1/len(solutions))*np.sum(data.loc[cvec[i],sel])
                dbary = np.abs(data.loc[cvec[i],sel]-bary)
                ncvec[i] = data.index[np.argmin(dbary)]
                tmpsolutions[np.where(ncvec[i]==data.index)[0]]=i #Updating new barycenters
            else:
                ncvec[i] = data.index.tolist()[np.argsort(data.loc[cvec[i],])[1]]
                tmpsolutions[np.where(ncvec[i]==data.index)[0]]=i

        cvec = sorted(ncvec)
        if set(cvec) in visited:
            print("converge")
            break
        visited.add(tuple(cvec))
        iters=iters-1
        pbar.update(1)
        return(solutions,cvec)