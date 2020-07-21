from glob import glob
import os.path as osp
import pickle

def pickle_load(path):
    with open(path, 'rb') as fid:
        data_ = pickle.load(fid)
    return data_

gnd = pickle_load('oxford5k/gnd_roxford5k.pkl')
query_names = gnd['qimlist']
index_names = gnd['imlist']
query_paths = ['oxford5k/oxford5k_images/%s.jpg'%x for x in query_names]
index_paths = ['oxford5k/oxford5k_images/%s.jpg'%x for x in index_names]

# jpgs = sorted(glob('oxford5k/oxford5k_images/*.jpg',  recursive=True))
with open('oxford5k/query_images.txt', 'w') as f:
    f.write('\n'.join(query_paths))
with open('oxford5k/index_images.txt', 'w') as f:
    f.write('\n'.join(index_paths))