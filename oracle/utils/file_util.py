import gzip
import pickle

def load_gzip(gzip_path: str):
    """Load gzip file (typically, PatchedPhaseDiagram).

    gzip_path(str): absolute path to gzip file.
    """
    with gzip.open(gzip_path, 'rb') as f:
        data = pickle.load(f)
    return data

def load_pkl(pkl_path: str):
    """Load pkl file (typically, PatchedPhaseDiagram).

    pkl_path(str): absolute path to pickle file.
    """
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    return data