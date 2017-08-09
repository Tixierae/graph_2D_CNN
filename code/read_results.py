import os
import json
import numpy as np

path_to_results = './results/'

results_names = os.listdir(path_to_results)

my_prec = 2 # desired precision

for name in results_names:
    with open(path_to_results + name, 'r') as my_file:
        tmp = json.load(my_file)
    vals = [elt[1] for elt in tmp['outputs']] # 'outputs' contains loss, accuracy for each repeat of each fold 
    vals = [val*100 for val in vals]
    print '=======',name,'======='
    print 'mean:', round(np.mean(vals),my_prec)
    print 'median:', round(np.median(vals),my_prec)
    print 'max:', round(max(vals),my_prec)
    print 'min:', round(min(vals),my_prec)
    print 'stdev', round(np.std(vals),my_prec)

    


    
