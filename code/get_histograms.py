import argparse
import os
import re
import numpy as np
import time as t
from multiprocessing import Pool, cpu_count
from functools import partial

# =============================================================================

parser = argparse.ArgumentParser()

# positional arguments (required)
parser.add_argument('path_to_node2vec', type=str, help='path to the root folder where node2vec arrays are stored for all datasets')
parser.add_argument('path_to_hist', type=str, help='path to the root folder where histograms should be written for all datasets')
parser.add_argument('dataset', type=str, help='name of the dataset. Must correspond to a valid value that matches names of files in node2vec folder')
parser.add_argument('p', type=str, help='p parameter of node2vec. Must correspond to a valid value that matches names of files in node2vec folder')
parser.add_argument('q', type=str, help='q parameter of node2vec. Must correspond to a valid value that matches names of files in node2vec folder')
parser.add_argument('definition', type=int, help='definition. E.g., 14 for 14:1. Must correspond to a valid value that matches names of files in node2vec folder')
parser.add_argument('max_n_channels', type=int, help='maximum number of channels that we will be able to pass to the network. Must not exceed half the depth of the tensors in node2vec folder')

args = parser.parse_args()

# convert command line arguments
path_to_node2vec = args.path_to_node2vec
path_to_hist = args.path_to_hist
dataset = args.dataset
p = args.p
q = args.q
definition = args.definition
max_n_channels = args.max_n_channels

# command line example: python get_histograms.py /home/antoine/Desktop/graph_2D_CNN/datasets/raw_node2vec/ /home/antoine/Desktop/graph_2D_CNN/datasets/tensors/ imdb_action_romance 1 1 14 5

# =============================================================================

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def get_hist_node2vec(emb,d,my_min,my_max,definition):
    # d should be an even integer
    img_dim = int(np.arange(my_min, my_max+0.05,(my_max+0.05-my_min)/float(definition*(my_max+0.05-my_min))).shape[0]-1)
    my_bins = np.linspace(my_min,my_max,img_dim) #  to have middle bin centered on zero
    Hs = []
    for i in range(0,d,2):
        H, xedges, yedges = np.histogram2d(x=emb[:,i],y=emb[:,i+1],bins=my_bins, normed=False)
        Hs.append(H)
    Hs = np.array(Hs)    
    return  Hs

def to_parallelize(my_file_name,dataset,n_dim,my_min,my_max,my_def,path_read,path_write):
    path_write_dataset = path_write + dataset + '/node2vec_hist/'
    
    p_value = [elt for elt in my_file_name.split('_') if elt.startswith('p=')][0]
    q_value = [elt for elt in my_file_name.split('_') if elt.startswith('q=')][0]
    real_idx =  my_file_name.split('.npy')[0].split('_')[-1:][0]
    emb = np.load(path_read + my_file_name)
    emb = emb[:,:n_dim]
    my_hist = get_hist_node2vec(emb=emb,d=n_dim,my_min=my_min,my_max=my_max,definition=my_def) 

    np.save(path_write_dataset + dataset + '_' + str(my_def) + ':1'+ '_' + p_value + '_' + q_value + '_' + real_idx, my_hist, allow_pickle=False)
    if int(real_idx) % 1000 == 0:
        print 'done', my_hist.shape

# =============================================================================

def main():
    t_start = t.time()
    
    n_dim = 2*max_n_channels
    
    all_file_names  = os.listdir(path_to_node2vec + dataset + '/') 
    print '===== total number of files in folder: =====', len(all_file_names)

    file_names_filtered = [elt for elt in all_file_names if dataset in elt and 'p=' + p in elt and 'q=' + q in elt]
    file_names_filtered.sort(key=natural_keys)
    print '===== number of files after filtering: =====', len(file_names_filtered)
    print '*** head ***'
    print file_names_filtered[:5]
    print '*** tail ***'
    print file_names_filtered[-5:]
    
    # load tensors
    tensors = []
    for idx, name in enumerate(file_names_filtered):
        tensor = np.load(path_to_node2vec + dataset + '/' + name)
        tensors.append(tensor[:,:n_dim])
        if idx % round(len(file_names_filtered)/10) == 0:
            print idx
    
    print 'tensors loaded'
    
    full = np.concatenate(tensors)
    my_max = np.amax(full)
    my_min = np.amin(full)
    print 'range:', my_max, my_min
    
    to_parallelize_partial = partial(to_parallelize, dataset=dataset, n_dim=n_dim, my_min=my_min, my_max=my_max, my_def=definition, path_read=path_to_node2vec + dataset + '/',path_write=path_to_hist)
    
    n_jobs = 2*cpu_count()

    print 'creating', n_jobs, 'jobs'
    
    pool = Pool(processes=n_jobs)
    pool.map(to_parallelize_partial, file_names_filtered)
    pool.close()

    print 'done in ', round(t.time() - t_start,4)

if __name__ == "__main__":
    main()