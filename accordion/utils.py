from accordion import distances as dst
import os
import pandas as pd
import os
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
import rpy2.robjects.packages as rpackages
from rpy2.robjects import pandas2ri
from sklearn.cluster import KMeans
from sklearn import metrics
import networkx as nx

def parse_CrossTalkeR(path):
	"""
	Parameters
    ----------
    path: Location to the RDS object generated by CrossTalkeR 
    
    Returns
    -------
    tbldata : DataFrame Containing all the Results from CrossTalkeR

    Notes
    -----
    This algorithm was proposed in [1]_ and [2]_.

	"""
	assert(os.path.exists(path) == True, "404: File Not Found")
	d = {'package.dependencies': 'package_dot_dependencies',
     'package_dependencies': 'package_uscore_dependencies'}
	ctker = importr('CrossTalkeR', 
                   robject_translations = d)
	readRDS = ro.r['readRDS']
	df = readRDS(path)
	phdat = df.slots['tables']
	tbldata = {}
	for i in enumerate(phdat.names):
	    with (ro.default_converter + pandas2ri.converter).context():
	        tbldata[i[1]] = pandas2ri.conversion.get_conversion().rpy2py(phdat[i[0]])
	return(tbldata)

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
			auxdict[f'{i}_{j}'] = 0.0
	p={}
	for k,v in graphs.items():
		p[k] = auxdict.copy()
		for ix, iy in dict(nx.get_edge_attributes(graphs[k],weight)).items():
			p[k]['_'.join(ix)] = iy         
	#for u in nodes:
	#	if u not in graphs[k].nodes():
	#			graphs[k].add_node(u)
	p = pd.DataFrame.from_dict(p)
	return(p)

def performance_eval(dist,y):
	average_linkage = KMeans(n_clusters=len(y.unique()),n_init='auto').fit_predict(dists)
	ars = metrics.adjusted_rand_score(average_linkage,y)
	silh = metrics.silhouette_score(dist,y)
	return {'ARS':ars, "Silhouette":silh} 