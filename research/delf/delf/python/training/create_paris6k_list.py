from glob import glob
import os.path as osp
import pickle

def pickle_load(path):
    with open(path, 'rb') as fid:
        data_ = pickle.load(fid)
    return data_

gnd = pickle_load('paris6k/gnd_rparis6k.pkl')
query_names = gnd['qimlist']
index_names = gnd['imlist']
query_paths = ['paris6k/paris6k_images/%s.jpg'%x for x in query_names]
index_paths = ['paris6k/paris6k_images/%s.jpg'%x for x in index_names]

with open('paris6k/query_images.txt', 'w') as f:
    f.write('\n'.join(query_paths))
with open('paris6k/index_images.txt', 'w') as f:
    f.write('\n'.join(index_paths))