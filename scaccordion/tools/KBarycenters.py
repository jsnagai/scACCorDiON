
import numpy as np 
import ot 
import pandas as pd

class KBarycenters:
    def __init__(self,k=3, max_iters=100,init='random',random_state=42):
        self.k = k
        self.max_iters = max_iters
        self.flabels = None
        self.bary = []
        self.centroids = None
        self.err = []
    
    def fit(self, data, distr,cost,init='random',random_state=42,reg=1e-2):
        np.random.seed(random_state)
        self._initialize_centroids(data,init,random_state)
        self.data = data
        for _ in range(self.max_iters):
            # Assign points to clusters
            self._assign_clusters()
            # Update centroids
            new_centroids = self._update_centroids(data, distr,cost,reg=reg)
            # Check for convergence
            if np.allclose(self.centroids, new_centroids):
                break
            self.centroids = new_centroids
     
    def _initialize_centroids(self, data, init,random_state):
        indices = []
        if init=='random':
            indices = np.random.choice(len(data.columns), self.k, replace=False)
            self.centroids = data.iloc[:,indices]
        elif init == '++':
            indices = [np.random.choice(len(data.columns),1, replace=False)]
            for idi in range(self.k-1):
                for idj in inidices:
                    indices.append(np.argmax(data[:,idj]))
            self.centroids = data.iloc[:,indices]

    def _assign_clusters(self):
        self.flabels = np.argmin(self.centroids, axis=1)
       
    def _update_centroids(self, data, distr,cost,reg=1e-2):
            tmpbary = {}
            tmperr = {}
            tmpdist = {}
            for i in range(self.k):
                currg = self.flabels==i
                bary= ot.barycenter(A=distr.loc[:,currg].apply(lambda x: x/sum(x)).to_numpy(),
                                     M=cost,reg=reg,log=True)
                tmperr[i] = np.median(bary[1]['err'])
                tmpbary[i] = bary[0]
                tmpdist[i]={}
                for idp in distr.columns:
                    tmpdist[i][idp] = ot.emd2(a=tmpbary[i]/tmpbary[i].sum(),
                                              b=distr[idp].to_numpy()/distr[idp].sum(),
                                              M=cost)
            self.bary=tmpbary
            tmpdist=pd.DataFrame.from_dict(tmpdist)
            for err in tmpdist.mean(axis=1):
                self.err.append(err)
            return tmpdist
 	