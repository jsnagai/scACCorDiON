import numpy as np
import numpy.linalg as lg
import networkx as nx
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import ot
from scipy import stats
from scipy import interpolate
from scipy.spatial.distance import pdist,squareform
import scipy.linalg as slg
import sklearn.metrics as measu
from sklearn import covariance
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import MDS
import logging as log
from accordion import utils as ault
from accordion import distances as acdist
from tqdm import tqdm

from pydiffmap import diffusion_map as dm

class Accordion():
	def __init__(self,tbls,weight='lr_means'):
		"""
		Build tCrossTalkeR Object

		Parameters
		----------
		tbls: tables of lr interactions

		"""
		self.tbls = tbls
		self.graphs = {}
		self.linegraphs = {}
		self.lgtable={}
		self.wdist={}
		self.Cs={}
		self.e = []
		self.logger = log.getLogger()
		self.logger.setLevel(log.DEBUG)
		self.history = {}
		self.expgraph = None
		self.perf ={}
		self.nodes = set()
		tmpcols = ['source','target',weight]
		for k,v in self.tbls.items():
			tmpvar = v.loc[:,tmpcols].groupby(['source','target']).sum().reset_index()
			self.graphs[k]=nx.from_pandas_edgelist(tmpvar,
		               							edge_attr=weight,
		               							create_using=nx.DiGraph)
			self.nodes.update(list(self.graphs[k].nodes()))
		self.p = ault.graphs_to_pmat(self.nodes,self.graphs,weight)
		self.history['Step1:']='Networkx build with success'

	def make_pca(self):
		"""
        Perform the principal component analysis using the matrix P
        
        Parameters
        ----------
        """
		pca = PCA()
		pca = pca.fit(self.p)
		pcawdist= pd.DataFrame.from_records(pca.components_).T
		pcawdist.index = self.p.T.index
		self.Cs['PCA'] = pcawdist
		self.history['Step2:']='PCA done!!!'
	
	def compute_cost_all(self):
		modes = ['CTD',
				 ('distance','correlation'),
				 'glasso']
		for i in modes:
			print(i)
			if type(i) is tuple:
				self.compute_cost(mode=i[0],metric=i[1])
			else:
				self.compute_cost(mode=i)
		self.history['Step3:']='Cost Computed'

	def compute_cost(self,mode='CTD',metric=None):
		"""
		Compute costs for the optimal transport

		Parameters
		----------
		mode:  str or function, optional
		The distance metric to use. The distance function can
		be 'distance','distancePCA','CTD','glasso'
		"""
		from scipy.spatial.distance import _METRICS
		if mode == 'distance':
			if metric in _METRICS.keys():
				self.Cs[metric]=squareform(pdist(self.p.to_numpy(),metric))
		elif mode == 'distancePCA':
			if metric in _METRICS.keys():
				# assert("PCA" in self.Cs,"PCA not computed")
				self.Cs[metric]=squareform(pdist(self.Cs['PCA'],metric))
		elif mode == 'CTD':
			self.expgraph = nx.DiGraph()
			for i in  self.p.index:
				tmpi = i.split("_")
				for j in  self.p.index:
					tmpj = j.split("_")
					if tmpi[1] == tmpj[0]:
						score = sum(( self.p.loc[i,] != 0) & ( self.p.loc[j,] != 0))
						score=((len(( self.p.loc[i,]))+1) - score)
						self.expgraph.add_edge(i,j,weight=score)
					elif tmpi[0] == tmpi[1]:
						score = sum(( self.p.loc[i,] != 0) & ( self.p.loc[j,] != 0))
						score=((len(( self.p.loc[i,]))+1) - score)
						self.expgraph.add_edge(j,i,weight=score)
			self.Cs['CTD']=acdist.ctd_dist(self.expgraph)
		elif mode == 'glasso':
			tmp = self.p.T/self.p.T.std(axis=0)
			glasso = covariance.GraphicalLassoCV(cv=10)
			glasso.fit(tmp)
			self.Cs['glasso'] = glasso.precision_
		else: 
			print('option not found')
        

    	
	def compute_wassestein(self,mode='single',cost='CTD',algorithm='emd',**kwargs):
		"""
		Compute Optimal Transport

		Parameters
		----------
		"""
		lab=cost
		if 'reg' in kwargs.keys():
			lab="f{cost}_{kwargs['reg']}"
		self.wdist[lab]={}
		if algorithm=='emd':
			for i in tqdm(self.p.columns):
				self.wdist[lab][i]={}
				for j in self.p.columns:
					self.wdist[lab][i][j] = ot.emd2(self.p[i].to_numpy()/self.p[i].sum(), 
													self.p[j].to_numpy()/self.p[j].sum(), 
													self.Cs[cost])
			self.wdist[lab] = pd.DataFrame.from_dict(self.wdist[lab])
		elif algorithm=='sinkhorn':
			for i in tqdm(self.p.columns):
			    for j in self.p.columns:
			    	self.wdist[lab][i][j] = ot.sinkhorn2(self.p[i].to_numpy()/self.p[i].sum(), 
			    										 self.p[j].to_numpy()/self.p[j].sum(), self.Cs[cost])
			self.wdist[lab] = pd.DataFrame.from_dict(self.wdist[lab])

	def eval_all(self,y):
		for lab in wdist.keys():
			ault.performance_eval(self.wdist[lab],y)




