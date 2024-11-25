import os
import pandas as pd
from sklearn import metrics
import networkx as nx
import kmedoids
from genieclust import compare_partitions

def graphs_to_pmat(nodes,graphs,weight='weight'):
	"""
	Parameters
    ----------
    nodes: Cell Cell Communication Node Universe
    graphs: Samples Cell Cell Communication 
    weight: weight
    Returns
    -------
    tbldata : DataFrame Containing all the Results from CrossTalkeR

    Notes 
    -----
    This algorithm was proposed in [1]_ and [2]_.

	"""
	auxdict = {}
	for i in nodes:
		for j in nodes:
			auxdict[f'{i}${j}'] = 0.0
	p={}
	for k,v in graphs.items():
		p[k] = auxdict.copy()
		for ix, iy in dict(nx.get_edge_attributes(graphs[k],weight)).items():
			p[k]['$'.join(ix)] = iy         
	p = pd.DataFrame.from_dict(p)
	p.fillna(0,inplace=True)
	return(p)

def performance_eval(dist,y):
	km = kmedoids.KMedoids(n_clusters=len(set(y)), method='fasterpam')
	c = km.fit(dist).labels_
	return compare_partitions.compare_partitions(c,y) 
