import pkg_resources
from .utils import parse_CrossTalkeR


def load_peng2019():
    """

    Return a dict with the LR pairs from the PDAC patients Described in Peng 2019
    
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    stream = pkg_resources.resource_filename(__name__, 'data/peng_ctker.Rds')
    return parse_CrossTalkeR(stream)



def load_peng2019_metadata():
    """

    Return a dict with the LR pairs from the PDAC patients Described in Peng 2019
    
    """
    # This is a stream-like object. If you want the actual info, call
    # stream.read()
    import pandas as pd
    stream = pkg_resources.resource_filename(__name__, 'data/Peng_PDAC_metaacc.pickle')
    return pd.read_pickle(stream)