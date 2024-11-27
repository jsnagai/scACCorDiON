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
from tqdm import tqdm
from pydiffmap import diffusion_map as dm
from .utils import *
from .distances import *


class Accordion():
	def __init__(self,tbls,weight='lr_means',filter=0.2,filter_mode='edge', normf=None, pseudo=1e-10,cost_new=False):
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
		# Convert graphs to vectors
		for k,v in self.tbls.items():
			tmpvar = v.loc[:,tmpcols].groupby(['source','target']).sum().reset_index()
			prop = v.loc[:,tmpcols].groupby(['source','target']).size().reset_index()
			prop[0]=prop[0]/prop[0].sum()
			tmpvar['prop'] =  prop[0]
			self.graphs[k]=nx.from_pandas_edgelist(tmpvar,
		               							edge_attr=[weight,'prop'],
		               							create_using=nx.DiGraph)
			self.nodes.update(list(self.graphs[k].nodes()))
		self.p = graphs_to_pmat(self.nodes,self.graphs,weight)
		self.p.sort_index(inplace=True)
		
		self.c = graphs_to_pmat(self.nodes,self.graphs,'prop')
		
		if normf != None:
			self.p = normf(self.p)
		# Filtering steps if needed
		if filter and filter_mode=='edge':
			self.p=self.p.loc[self.p.T.var() > np.quantile(self.p.T.var(),q=filter),:]
		elif filter and filter_mode=='sample':
			self.p=self.p.loc[:,self.p.var() > np.quantile(self.p.var(),q=filter)]
		elif filter and filter_mode=='both':
			self.p=self.p.loc[self.p.T.var() > np.quantile(self.p.T.var(),q=filter),:]
			self.p=self.p.loc[:,self.p.var() > np.quantile(self.p.var(),q=filter)]
		# Creating common grounded graph
		if cost_new:
			self.p[self.p<pseudo]=0
			self.p = self.p.loc[self.p.sum(axis=1)!=0,:]
			esize = len(self.p.index)
			tmpmat = (esize+1)*np.ones((esize,esize))
			lmax = 0
			for i in enumerate(self.p.index):
				tmpi = i[1].split("$")
				for j in  enumerate(self.p.index):
					tmpj = j[1].split("$")
					if i!=j:
						if tmpi[1] == tmpj[0] or tmpi[0] == tmpj[1]:  #give u,v edges check tail u is head v
							score = sum(( self.p.iloc[i[0],:] <=pseudo) & ( self.p.iloc[j[0],:] != pseudo))
							score=((len(( self.p.iloc[i[0],:]))+1) - score)
							tmpmat[i[0]][j[0]]=score
							if score != esize+1 and score>lmax:
								lmax=score
			self.expgraph = pd.DataFrame(tmpmat,index=self.p.index,columns=self.p.index)
			#self.expgraph.replace(to_replace=esize+1,value=0,inplace=True)
			self.e = self.expgraph 
			self.expgraph = nx.from_pandas_adjacency(1-self.expgraph.apply(lambda x:x/(sum(x)+pseudo),axis=1) ,create_using=nx.DiGraph)
			self.history['Step1:']='Networkx build with success'
		else:
			# Creating common grounded graph
			self.expgraph = nx.DiGraph()
			self.p = self.p.loc[self.p.sum(axis=1)!=0,:]
			for i in self.p.index:
				tmpi = i.split("$")
				for j in  self.p.index:
					tmpj = j.split("$")
					if i!=j:
						if tmpi[1] == tmpj[0] or tmpi[0] == tmpj[1]:  #give u,v edges check tail u is head v
							score = sum(( self.p.loc[i,] > pseudo) & ( self.p.loc[j,] > pseudo))
							score=((len(( self.p.loc[i,]))+1) - score)
							self.expgraph.add_edge(i,j,weight=score/len(self.p.columns))
			self.e = nx.to_pandas_adjacency(self.expgraph) 
			tmpdf=nx.to_pandas_adjacency(self.expgraph).apply(lambda x:x/(sum(x)+pseudo),axis=1) 
			self.expgraph = nx.from_pandas_adjacency(tmpdf,create_using=nx.DiGraph)
			self.history['Step1:']='Networkx build with success'


	def make_pca(self):
		"""
        Perform the principal component analysis using the matrix P
        
        Parameters
        ----------
        """
		pca = PCA(svd_solver='full')
		pca = pca.fit(self.p)
		pcawdist= pd.DataFrame.from_records(pca.components_).T
		pcawdist.index = self.p.T.index
		self.Cs['PCA'] = pcawdist
		self.history['Step2:']='PCA done!!!'
	
	def compute_cost_all(self):
		modes = ['GRD',
				 ('distance','correlation'),
				 'glasso',('HTD',0.5),('HTD',1)]
		for i in modes:
			print(i)
			if type(i) is tuple:
				if i[0]=='distance':
					self.compute_cost(mode=i[0],metric=i[1])
				else:
					self.compute_cost(mode=i[0],beta=i[1])
			else:
				self.compute_cost(mode=i)
		self.history['Step3:']='Cost Computed'

	def compute_cost(self,mode='GRD',metric=None,beta=0.5,d=1e-10,degnorm=False):
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
				self.wdist[f'PCA_{metric}']=squareform(pdist(self.Cs['PCA'],metric))
		elif mode == 'GRD':
			self.Cs['GRD']=getGRD(self.expgraph,degnorm=False)
		elif mode == 'HTD':
			tmp = nx.to_numpy_array(self.expgraph)
			tmp += d*np.ones(tmp.shape)
			self.Cs[f'HTD_{beta}'] = getCTD(tmp,beta=beta)
		else: 
			print('option not found')
        

    	
	def compute_wassestein(self,mode='single',cost='GRD',algorithm='emd',**kwargs):
		"""
		Compute Optimal Transport

		Parameters
		----------
		"""
		lab=cost
		if 'reg' in kwargs.keys():
			tmpl = kwargs["reg"]
			lab=f'{cost}_{tmpl}'
		if 'reg_m' in kwargs.keys():
			tmpl = kwargs["reg"]
			if len(kwargs["reg_m"]) > 1:
				marg = '_'.join([str(i) for i in kwargs["reg_m"]])
			else:
				marg = str(kwargs["reg_m"])
			lab=f'{cost}_{tmpl}:{marg}'
		self.wdist[lab]={}
		if algorithm=='emd':
			for i in self.p.columns:
				self.wdist[lab][i]={}
				for j in self.p.columns:
					self.wdist[lab][i][j] = ot.emd2(a=self.p[i].to_numpy()/self.p[i].sum(), 
													b=self.p[j].to_numpy()/self.p[j].sum(), 
													M=self.Cs[cost])
			self.wdist[lab] = pd.DataFrame.from_dict(self.wdist[lab])
		elif algorithm=='sinkhorn':
			for i in self.p.columns:
				self.wdist[lab][i]={}
				for j in self.p.columns:
					self.wdist[lab][i][j] = ot.sinkhorn2(self.p[i].to_numpy()/self.p[i].sum(), 
			    										 self.p[j].to_numpy()/self.p[j].sum(), 
			    										 self.Cs[cost],**kwargs)
			self.wdist[lab] = pd.DataFrame.from_dict(self.wdist[lab])
		elif algorithm =='unbalanced':
			for i in self.p.columns:
				self.wdist[lab][i]={}
				for j in self.p.columns:
					self.wdist[lab][i][j] = ot.unbalanced.sinkhorn_unbalanced2(self.p[i].to_numpy()/self.p[i].sum(), 
			    										 self.p[j].to_numpy()/self.p[j].sum(), 
			    										 self.Cs[cost],**kwargs)
			self.wdist[lab] = pd.DataFrame.from_dict(self.wdist[lab])
	def eval_all(self,y):
		tmpeval = {}
		for lab in self.wdist.keys():
			tmpeval[lab]=performance_eval(self.wdist[lab].to_numpy(),y)
		return tmpeval