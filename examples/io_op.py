import pickle

def write_pkl(content, path):
    with open(path, 'wb') as f:
        pickle.dump(content, f)

def read_pkl(path):
    with open(path, 'rb') as f:
        return pickle.load(f)