import pickle
from pathlib import Path

# TODO save_ppfl
def OrderedDict_to_pkl(Odict, filename, save_dir=None):
    if save_dir != None:
        with open(Path(save_dir,filename), 'wb') as f:
            pickle.dump(Odict, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(filename, 'wb') as f:
            pickle.dump(Odict, f, pickle.HIGHEST_PROTOCOL)

def pkl_to_OrderedDict(filename, save_dir=None):
    if save_dir != None:
        with open(Path(save_dir,filename), 'rb') as f:
            Odict = pickle.load(f)
    else:
        with open(filename, 'rb') as f:
            Odict = pickle.load(f)
    return Odict


def save_pickle(data, filename):
    with open(filename, 'wb') as f:
        pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)

def load_pickle(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data