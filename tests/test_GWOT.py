import scaccordion as actl
import pytest
import pandas as pd
import networkx as nx
import copy
import numpy as np

def test_case1_github():
	k = 30;
	A = np.random.rand(k,k)
	mA= (1/k)*np.ones(k)
	res =  actl.tl.GWOT.emd2RTLB(A,A,mA,mA)
	assert res[0] == 0

def test_case2_github():
	k = 40
	A = np.random.rand(k,k)
	mA= (1/k)*np.ones(k)
	B = 10*np.random.rand(k,k) + 10
	mB= (1/k)*np.ones(k)
	res = actl.tl.GWOT.emd2RTLB(A,B,mA,mB)
	revres = actl.tl.GWOT.emd2RTLB(B,A,mB,mA)
	assert pytest.approx(res[0],1e-4) == revres[0]