import scaccordion as actl
import pytest
import pandas as pd
import networkx as nx
import copy

def test_got():
	n = 14
	g1 = nx.stochastic_block_model([7,7],[[0.9,0.1],[0.1,0.9]], seed = 8576)
	l1 = nx.laplacian_matrix(g1,range(n))
	l1 = l1.todense()
	g2 = copy.deepcopy(g1)
	g2.remove_edge(4,13)
	g2.remove_edge(3,7)
	l2 = nx.laplacian_matrix(g2,range(n))
	l2 = l2.todense()
	g3 = copy.deepcopy(g1)
	g3.remove_edge(4,6)
	g3.remove_edge(8,9)
	l3 = nx.laplacian_matrix(g3,range(n))
	l3 = l3.todense()
	res = [actl.tl.GOT.wass_dist_(l1,l2),actl.tl.GOT.wass_dist_(l1,l3)]
	assert pytest.approx(res,0.005) == [0.9123,0.0134]