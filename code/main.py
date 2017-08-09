import argparse

import math
import numpy as np
import os
import random
import json
import re
import datetime
import time

from keras.layers import Dense, Dropout, Flatten, Input, Convolution2D, MaxPooling2D, Merge
from keras.utils import np_utils
from keras.models import Model
from keras import backend as K

from keras.callbacks import EarlyStopping

# =============================================================================

parser = argparse.ArgumentParser()

# positional arguments (required)
parser.add_argument('path_root', type=str, help="path to 'datasets' directory")
parser.add_argument('dataset', type=str, help='name of the dataset. Must correspond to a valid value that matches names of files in tensors/*dataset*/node2vec_hist/ folder')
parser.add_argument('p', type=str, help='p parameter of node2vec. Must correspond to a valid value that matches names of files in tensors/*dataset*/node2vec_hist/ folder')
parser.add_argument('q', type=str, help='q parameter of node2vec. Must correspond to a valid value that matches names of files in tensors/*dataset*/node2vec_hist/ folder')
parser.add_argument('definition', type=int, help='definition. E.g., 14 for 14:1. Must correspond to a valid value that matches names of files in tensors/*dataset*/node2vec_hist/ folder')
parser.add_argument('n_channels', type=int, help='number of channels. Must not exceed half the depth of the tensors in tensors/*dataset*/node2vec_hist/ folder')

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

n_folds = args.n_folds
n_repeats = args.n_repeats
batch_size = args.batch_size
nb_epochs = args.nb_epochs
my_patience = args.patience
drop_rate = args.drop_rate

dim_ordering = 'th' # channels first
my_optimizer = 'adam'

# command line examples: python main.py /home/antoine/Desktop/graph_2D_CNN/datasets/ imdb_action_romance 1 1 14 5
#                        python main.py /home/antoine/Desktop/graph_2D_CNN/datasets/ imdb_action_romance 1 1 14 5 --n_folds 10 --n_repeats 1 --nb_epochs 20 --patience 3

# =============================================================================

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    return [atoi(c) for c in re.split('(\d+)', text)]

# =============================================================================

def main():
    
    my_date_time = '_'.join(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S").split())

    parameters = {'path_root':path_root,
                  'dataset':dataset,
                  'p':p,
                  'q':q,
                  'definition':definition,
                  'n_channels':n_channels,
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

    print '========== loading tensors =========='

    path_read = path_root + 'tensors/' + dataset + '/node2vec_hist/'
    file_names = [elt for elt in os.listdir(path_read) if (str(definition)+':1' in elt and regexp_p.search(elt) and regexp_q.search(elt) and elt.count('p=')==1 and elt.count('q=')==1 and elt.split('_')[-1:][0][0].isdigit())] # make sure the right files are selected
    file_names.sort(key=natural_keys)
    print len(file_names)
    print file_names[:5]
    print file_names[-5:]

    print 'ensuring tensor-label matching'
    kept_idxs = [int(elt.split('_')[-1].split('.')[0]) for elt in file_names]
    print len(kept_idxs)
    print kept_idxs[:5]
    print kept_idxs[-5:]
    print 'removing', len(ys) - len(kept_idxs), 'labels'
    ys = [y for idx,y in enumerate(ys) if idx in kept_idxs]

    print len(file_names) == len(ys)

    print 'converting labels to array'
    ys = np.array(ys)

    print 'transforming integer labels into one-hot vectors'
    ys = np_utils.to_categorical(ys, num_classes)

    tensors = []
    for name in file_names:
        tensor = np.load(path_read + name)
        tensors.append(tensor[:n_channels,:,:])

    tensors = np.array(tensors)
    tensors = tensors.astype('float32')

    print 'tensors shape:', tensors.shape

    print '========== getting image dimensions =========='

    img_rows, img_cols = int(tensors.shape[2]), int(tensors.shape[3])
    input_shape = (int(tensors.shape[1]), img_rows, img_cols)   

    print 'input shape:', input_shape 

    print '========== shuffling data =========='

    shuffled_idxs = random.sample(range(tensors.shape[0]), int(tensors.shape[0])) # sample w/o replct
    tensors = tensors[shuffled_idxs]
    ys = ys[shuffled_idxs]
      
    print '========== conducting', n_folds ,'fold cross validation =========='; print 'repeating each fold:', n_repeats, 'times'

    folds = np.array_split(tensors,n_folds,axis=0)

    print 'fold sizes:', [len(fold) for fold in folds]

    folds_labels = np.array_split(ys,n_folds,axis=0)

    outputs = []
    histories = []

    for i in range(n_folds):
        
        t = time.time()
        
        x_train = np.concatenate([fold for j,fold in enumerate(folds) if j!=i],axis=0)
        x_test = [fold for j,fold in enumerate(folds) if j==i]
        
        y_train = np.concatenate([y for j,y in enumerate(folds_labels) if j!=i],axis=0)
        y_test = [y for j,y in enumerate(folds_labels) if j==i]
        
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
            
            history = model.fit(x_train,
                                y_train,
                                batch_size=batch_size,
                                nb_epoch=nb_epochs,
                                validation_data=(x_test, y_test),
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
