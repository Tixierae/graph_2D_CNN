import argparse

import math
import numpy as np
import os
import random
import json
import re
import datetime
import time
from multiprocessing import Pool, cpu_count
from functools import partial

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KernelDensity

from keras.layers import Dense, Dropout, Flatten, Input, Convolution2D, MaxPooling2D, Merge
from keras.utils import np_utils
from keras.models import Model
from keras import backend as K
from keras.callbacks import EarlyStopping

# =============================================================================

parser = argparse.ArgumentParser()

# positional arguments (required)
parser.add_argument('path_root', type=str, help="path to 'datasets' directory")
parser.add_argument('dataset', type=str, help='name of the dataset. Must correspond to a valid value that matches names of files in node2vec folder')
parser.add_argument('p', type=str, help='p parameter of node2vec. Must correspond to a valid value that matches names of files in node2vec folder')
parser.add_argument('q', type=str, help='q parameter of node2vec. Must correspond to a valid value that matches names of files in node2vec folder')
parser.add_argument('definition', type=int, help='definition. E.g., 14 for 14:1. Must correspond to a valid value that matches names of files in node2vec folder')
parser.add_argument('n_channels', type=int, help='number of channels that we will be passed to the network. Must not exceed half the depth of the tensors in node2vec folder')
parser.add_argument('n_bootstrap', type=float, help='augmentation ratio. Must be strictly between 0 and 1')

# optional arguments
parser.add_argument('--n_folds', type=int, default=10, choices=[2,3,4,5,6,7,8,9,10], help='number of folds for cross-validation')
parser.add_argument('--n_repeats', type=int, default=3, choices=[1,2,3,4,5], help='number of times each fold should be repeated')
parser.add_argument('--batch_size', type=int, default=32, choices=[32,64,128], help='batch size')
parser.add_argument('--nb_epochs', type=int, default=50, help='maximum number of epochs')
parser.add_argument('--patience', type=int, default=5, help='patience for early stopping strategy')
parser.add_argument('--drop_rate', type=float, default=0.3, help='dropout rate')

args = parser.parse_args()

# convert command line arguments
path_root = args.path_root
dataset = args.dataset
p = args.p
q = args.q
definition = args.definition
n_channels = args.n_channels
n_bootstrap = args.n_bootstrap

n_folds = args.n_folds
n_repeats = args.n_repeats
batch_size = args.batch_size
nb_epochs = args.nb_epochs
my_patience = args.patience
drop_rate = args.drop_rate

dim_ordering = 'th' # channels first
my_optimizer = 'adam'
params = {'bandwidth': np.logspace(-1, -0.5, 10)} # for the kernel bandwidth grid search

# command line example: python main_data_augmentation.py /home/antoine/Desktop/graph_2D_CNN/datasets/ imdb_action_romance 1 1 14 5 0.1

# =============================================================================

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

def get_bw_cv(x,params):
    grid = GridSearchCV(KernelDensity(), params,cv=2,n_jobs=1)
    grid.fit(x[:,None])
    bw = grid.best_estimator_.bandwidth
    return bw

def get_hist_node2vec(emb,d,my_min,my_max,definition):
    # d should be an even integer
    img_dim = int(np.arange(my_min, my_max+0.05,(my_max+0.05-my_min)/float(definition*(my_max+0.05-my_min))).shape[0]-1)
    my_bins = np.linspace(my_min,my_max,img_dim) #  to have middle bin centered on zero
    Hs = []
    for i in range(0,d,2):
        H, xedges, yedges = np.histogram2d(x=emb[:,i],y=emb[:,i+1],bins=my_bins, normed=False)
        Hs.append(H)
    Hs = np.array(Hs)    
    return Hs

def smoothed_bootstrap(my_array,params):
    # compute mean and variance along each dimension
    my_means = np.mean(my_array,0)
    my_vars = np.var(my_array,0)
    
    # to save time, estimate bandwidth from at most the first 100 nodes
    my_bws = np.apply_along_axis(get_bw_cv,0,my_array[:min(100,my_array.shape[0]),:],params)

    all_new_coords = []
    for jj in range(int(np.random.normal(my_array.shape[0],scale=my_array.shape[0]/5))):
        rand_row_idx = random.randint(0,my_array.shape[0]-1) # select a row index (i.e., a node) at random
        my_coords = my_array[rand_row_idx,:].tolist()
        new_coords = [0]*len(my_coords)
        for kk in range(len(new_coords)): # for each dim
            new_coords[kk] = my_means[kk] + (my_coords[kk] - my_means[kk] + np.random.normal(0,scale=(my_bws[kk])**0.5))/((1+my_bws[kk]/my_vars[kk])**(.5))
        all_new_coords.append(new_coords)
    all_new_coords = np.array(all_new_coords)    
    return all_new_coords

# =============================================================================

def main():
    
    my_date_time = '_'.join(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").split())

    parameters = {'path_root':path_root,
                  'dataset':dataset,
                  'p':p,
                  'q':q,
                  'definition':definition,
                  'n_channels':n_channels,
                  'n_bootstrap':n_bootstrap,
                  'n_folds':n_folds,
                  'n_repeats':n_repeats,
                  'batch_size':batch_size,
                  'nb_epochs':nb_epochs,
                  'my_patience':my_patience,
                  'drop_rate':drop_rate,
                  'dim_ordering':dim_ordering,
                  'my_optimizer':my_optimizer
                  }

    name_save = path_root + '/results/' + dataset + '_augmentation_' + my_date_time
      
    with open(name_save + '_parameters.json', 'w') as my_file:
        json.dump(parameters, my_file, sort_keys=True, indent=4)

    print '========== parameters defined and saved to disk =========='

    regexp_p = re.compile('p=' + p)
    regexp_q = re.compile('q=' + q)
    n_dim = 2*n_channels
    inverse_n_b = int(round(1/n_bootstrap))
    smoothed_bootstrap_partial = partial(smoothed_bootstrap, params=params) # for parallelization
    n_jobs = cpu_count()

    print '========== loading labels =========='

    with open(path_root + 'classes/' + dataset + '/' + dataset + '_classes.txt', 'r') as f:
        ys = f.read().splitlines()
        ys = [int(elt) for elt in ys]

    num_classes = len(list(set(ys)))

    print 'classes:', list(set(ys))

    print 'converting to 0-based index'

    if 0 not in list(set(ys)):
        if -1 not in list(set(ys)):
            ys = [y-1 for y in ys]
        else:
            ys = [1 if y==1 else 0 for y in ys]

    print 'classes:', list(set(ys))  

    print '========== loading node2vec embeddings =========='
    
    all_file_names  = os.listdir(path_root + '/raw_node2vec/' + dataset + '/') 
    print '===== total number of files in folder: =====', len(all_file_names)
    
    file_names_filtered = [elt for elt in all_file_names if (dataset in elt and regexp_p.search(elt) and regexp_q.search(elt) and elt.count('p=')==1 and elt.count('q=')==1 and elt.split('_')[-1:][0][0].isdigit())]
    file_names_filtered.sort(key=natural_keys)
    
    print 'number of files after filtering:', len(file_names_filtered)
    print '*** head ***'
    print file_names_filtered[:5]
    print '*** tail ***'
    print file_names_filtered[-5:]
    
    # load tensors
    raw_emb = []
    excluded_idxs = []
    for idx, name in enumerate(file_names_filtered):
        emb = np.load(path_root + '/raw_node2vec/' + dataset + '/' + name)
        if emb.shape[1]<n_dim:
            excluded_idxs.append(idx)
        else:
            raw_emb.append(emb[:,:n_dim])
        if idx % round(len(file_names_filtered)/10) == 0:
            print idx
    
    print 'node2vec embeddings loaded'
    
    print 'ensuring tensor-label matching 1st attempt'
    print 'removing', len(excluded_idxs), 'labels'
    ys = [y for idx,y in enumerate(ys) if idx not in excluded_idxs]
    print len(raw_emb) == len(ys)
    
    print 'ensuring tensor-label matching 2nd attempt'
    kept_idxs = [int(elt.split('_')[-1].split('.')[0]) for elt in file_names_filtered]
    print 'removing', len(ys) - len(kept_idxs), 'labels'
    ys = [y for idx,y in enumerate(ys) if idx in kept_idxs]
    print len(file_names_filtered) == len(ys)

    full = np.concatenate(raw_emb)
    my_max = np.amax(full)
    my_min = np.amin(full)
    print 'range:', my_max, my_min
    
    img_dim = int(np.arange(my_min, my_max+0.05,(my_max+0.05-my_min)/float(definition*(my_max+0.05-my_min))).shape[0]-1)
    print 'img_dim:', img_dim
    
    print '========== shuffling data =========='

    shuffled_idxs = random.sample(range(len(ys)), int(len(ys))) # sample w/o replct
    raw_emb = [raw_emb[idxx] for idxx in shuffled_idxs]
    ys = [ys[idxx] for idxx in shuffled_idxs]
      
    print '========== conducting', n_folds ,'fold cross validation =========='; print 'repeating each fold:', n_repeats, 'times'

    folds = np.array_split(raw_emb,n_folds,axis=0)

    print 'fold sizes:', [len(fold) for fold in folds]

    folds_labels = np.array_split(ys,n_folds,axis=0)

    outputs = []
    histories = []

    for i in range(n_folds):
    
        t = time.time()
        
        raw_emb_train = [fold for j,fold in enumerate(folds) if j!=i]
        raw_emb_train = [elt for sublist in raw_emb_train for elt in sublist] # flatten
        
        raw_emb_test = [fold for j,fold in enumerate(folds) if j==i][0]
        
        y_train = [y for j,y in enumerate(folds_labels) if j!=i]
        y_train = [elt for sublist in y_train for elt in sublist] # flatten
        y_test = [y for j,y in enumerate(folds_labels) if j==i][0]          
        
        print '*** resampling training set ***'
        print '* generating one bootstrapped sample for every', inverse_n_b ,'training obs *'
        
        idx_for_boot = random.sample(range(len(raw_emb_train)), int(round(n_bootstrap*len(raw_emb_train))))
            
        print 'creating', n_jobs, 'jobs'
        tt = time.time()
        
        pool = Pool(processes=n_jobs)
        new_raw_emb_train = pool.map(smoothed_bootstrap_partial, [elt for idx,elt in enumerate(raw_emb_train) if idx in idx_for_boot])
        pool.close()

        print len(idx_for_boot), 'bootstrap samples created in ', round(time.time() - tt,4)
            
        new_y_train = [elt for idx,elt in enumerate(y_train) if idx in idx_for_boot]
        
        # append original embeddings and labels
        new_raw_emb_train = new_raw_emb_train + raw_emb_train
        new_y_train = new_y_train + y_train

        print 'shuffling data and labels'
        shuffled_idxs = random.sample(range(len(new_raw_emb_train)), int(len(new_raw_emb_train))) # sample w/o replct
        new_raw_emb_train = [new_raw_emb_train[elt] for elt in shuffled_idxs]
        new_y_train = [new_y_train[elt] for elt in shuffled_idxs]
        
        print 'computing histograms on the training set'
        tensors_train = []
        for idx, my_new_emb in enumerate(new_raw_emb_train):
            tensors_train.append(get_hist_node2vec(emb=my_new_emb,d=n_dim,my_min=my_min,my_max=my_max,definition=definition)[:n_channels,:,:])
            if idx % round(len(new_raw_emb_train)/float(10)) == 0:
                print idx
        
        print 'computing histograms on the test set'
        tensors_test = []
        for my_emb in raw_emb_test:
            tensors_test.append(get_hist_node2vec(emb=my_emb,d=n_dim,my_min=my_min,my_max=my_max,definition=definition)[:n_channels,:,:])

        print 'converting labels to array'
        new_y_train = np.array(new_y_train)
        y_test = np.array(y_test)

        print 'transforming integer labels into one-hot vectors'
        new_y_train = np_utils.to_categorical(new_y_train, num_classes)
        y_test = np_utils.to_categorical(y_test, num_classes)
        
        print 'transforming tensors into numpy arrays'
        tensors_train = np.array(tensors_train)
        tensors_train = tensors_train.astype('float32')
        
        tensors_test = np.array(tensors_test)
        tensors_test = tensors_test.astype('float32')

        print 'tensors training shape:', tensors_train.shape
        print 'tensors test shape:', tensors_test.shape

        # input image dimensions
        img_rows, img_cols = int(tensors_train.shape[2]), int(tensors_train.shape[3])
        input_shape = (int(tensors_train.shape[1]), img_rows, img_cols)    
        print 'input shape:', input_shape 
            
        for repeating in range(n_repeats):
            
            print 'clearing Keras session'
            K.clear_session()
            
            my_input = Input(shape=input_shape, dtype='float32')
            
            conv_1 = Convolution2D(64,
                                   3,
                                   3,
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_1 = MaxPooling2D(pool_size=(2,2),
                                         dim_ordering=dim_ordering
                                         )(conv_1)

            pooled_conv_1_dropped = Dropout(drop_rate)(pooled_conv_1)
            
            conv_11 = Convolution2D(96,
                                    3,
                                    3,
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering
                                    )(pooled_conv_1_dropped)
            
            pooled_conv_11 = MaxPooling2D(pool_size=(2,2),
                                          dim_ordering=dim_ordering
                                          )(conv_11)
                                          
            pooled_conv_11_dropped = Dropout(drop_rate)(pooled_conv_11)
            pooled_conv_11_dropped_flat = Flatten()(pooled_conv_11_dropped)

            conv_2 = Convolution2D(64,
                                   4,
                                   4, 
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_2 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_2)
            pooled_conv_2_dropped = Dropout(drop_rate)(pooled_conv_2)
            
            conv_22 = Convolution2D(96,
                                    4,
                                    4, 
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering,
                                    )(pooled_conv_2_dropped)
            
            pooled_conv_22 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_22)
            pooled_conv_22_dropped = Dropout(drop_rate)(pooled_conv_22)
            pooled_conv_22_dropped_flat = Flatten()(pooled_conv_22_dropped)

            conv_3 = Convolution2D(64,
                                   5,
                                   5,
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_3 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_3)
            pooled_conv_3_dropped = Dropout(drop_rate)(pooled_conv_3)
            
            conv_33 = Convolution2D(96,
                                    5,
                                    5,
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering
                                    )(pooled_conv_3_dropped)
            
            pooled_conv_33 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_33)
            pooled_conv_33_dropped = Dropout(drop_rate)(pooled_conv_33)
            pooled_conv_33_dropped_flat = Flatten()(pooled_conv_33_dropped)                        
            
            conv_4 = Convolution2D(64,
                                   6,
                                   6,
                                   border_mode='valid',
                                   activation='relu',
                                   dim_ordering=dim_ordering
                                   )(my_input)
            
            pooled_conv_4 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering)(conv_4)
            pooled_conv_4_dropped = Dropout(drop_rate)(pooled_conv_4)
            
            conv_44 = Convolution2D(96,
                                    6,
                                    6,
                                    border_mode='valid',
                                    activation='relu',
                                    dim_ordering=dim_ordering
                                    )(pooled_conv_4_dropped)
            
            pooled_conv_44 = MaxPooling2D(pool_size=(2,2),dim_ordering=dim_ordering) (conv_44)
            pooled_conv_44_dropped = Dropout(drop_rate) (pooled_conv_44)
            pooled_conv_44_dropped_flat = Flatten()(pooled_conv_44_dropped)

            merge = Merge(mode='concat')([pooled_conv_11_dropped_flat,
                                          pooled_conv_22_dropped_flat,
                                          pooled_conv_33_dropped_flat,
                                          pooled_conv_44_dropped_flat])
            
            merge_dropped = Dropout(drop_rate)(merge)
            
            dense = Dense(128,
                          activation='relu'
                          )(merge_dropped)
            
            dense_dropped = Dropout(drop_rate)(dense)
            
            prob = Dense(output_dim=num_classes,
                         activation='softmax'
                         )(dense_dropped)
            
            # instantiate model
            model = Model(my_input,prob)
                            
            # configure model for training
            model.compile(loss='categorical_crossentropy',
                          optimizer=my_optimizer,
                          metrics=['accuracy'])
            
            print 'model compiled'
            
            early_stopping = EarlyStopping(monitor='val_acc', # go through epochs as long as acc on validation set increases
                                           patience=my_patience,
                                           mode='max') 
            
            history = model.fit(tensors_train,
                                new_y_train,
                                batch_size=batch_size,
                                nb_epoch=nb_epochs,
                                validation_data=(tensors_test, y_test),
                                callbacks=[early_stopping])
            
            # save [min loss,max acc] on test set
            max_acc = max(model.history.history['val_acc'])
            max_idx = model.history.history['val_acc'].index(max_acc)
            output = [model.history.history['val_loss'][max_idx],max_acc]
            outputs.append(output)
            
            # also save full history for sanity checking
            histories.append(model.history.history)
        
        print '**** fold', i+1 ,'done in ' + str(math.ceil(time.time() - t)) + ' second(s) ****'

    # save results to disk
    with open(name_save + '_results.json', 'w') as my_file:
        json.dump({'outputs':outputs,'histories':histories}, my_file, sort_keys=False, indent=4)

    print '========== results saved to disk =========='

if __name__ == "__main__":
    main()
