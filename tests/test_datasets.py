import scaccordion.tl as actl
import pytest
import pandas as pd

def test_pdac_load():
    assert type(actl.datasets.load_peng2019()) is dict

def test_pdac_metadata_load():
    assert type(actl.datasets.load_peng2019_metadata()) is pd.DataFrame

@pytest.mark.parametrize("datashape,shape",[(actl.datasets.load_peng2019_metadata().shape,(35,2))])
def test_pdac_metadata_load(datashape,shape):
    assert datashape == shape

def test_pdac_keys():
    import numpy as np
    df = actl.datasets.load_peng2019()
    metadata = actl.datasets.load_peng2019_metadata()
    assert np.all(metadata.index.isin(df.keys()))

def test_gl_plain_net():
    g1 = actl.datasets.make_gl_ciclic_graph()
    assert [len(g1.nodes()),len(g1.edges())] == [16,24]
