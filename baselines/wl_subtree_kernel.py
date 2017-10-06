"""
- Sample code for:
"Classifying Graphs as Images with Convolutional Neural Networks"
arXiv preprint arXiv:1708.02218

- Computes the Weisfeiler-Lehman subtree kernel with h iterations for a set of graphs. Each vertex is assigned its degree as label.

- Datasets should be placed into the "datasets" folder and can be downloaded from: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

- To run, use the following command: 
python wl_subtree_kernel.py dataset h

- Part of the code is modified from:
"Deep graph kernels"
Proceedings of the 21th International Conference on Knowledge Discovery and Data Mining, 2015.
"""

import networkx as nx
import numpy as np
import sys
from collections import defaultdict
import copy
	
np.random.seed(None)

def load_data(ds_name):
	node2graph = {}
	Gs = []
	
	with open("../datasets/%s/%s_graph_indicator.txt"%(ds_name,ds_name), "rb") as f:
		c = 1
		for line in f:
			node2graph[c] = int(line[:-1])
			if not node2graph[c] == len(Gs):
				Gs.append(nx.Graph())
			Gs[-1].add_node(c)
			c += 1
	
	with open("../datasets/%s/%s_A.txt"%(ds_name,ds_name), "rb") as f:
		for line in f:
			edge = line[:-1].split(",")
			edge[1] = edge[1].replace(" ", "")
			Gs[node2graph[int(edge[0])]-1].add_edge(int(edge[0]), int(edge[1]))
		
	labels = []
	with open("../datasets/%s/%s_graph_labels.txt"%(ds_name,ds_name), "rb") as f:
		for line in f:
			labels.append(int(line[:-1]))
	
	labels  = np.array(labels, dtype=np.float)
	return Gs, labels	


def wl_subtree_kernel(graphs, h):
	N = len(graphs)

	labels = {}
	label_lookup = {}
	label_counter = 0

	for G in graphs:
		for node in G.nodes():
			G.node[node]['label'] = G.degree(node)

	orig_graph_map = {it: {i: defaultdict(lambda: 0) for i in range(N)} for it in range(-1, h)}

	# initial labeling
	ind = 0
	for G in graphs:
		labels[ind] = np.zeros(G.number_of_nodes(), dtype = np.int32)
		node2index = {}
		for node in G.nodes():
		    node2index[node] = len(node2index)
		    
		for node in G.nodes():
		    label = G.node[node]['label']
		    if not label_lookup.has_key(label):
		        label_lookup[label] = len(label_lookup)

		    labels[ind][node2index[node]] = label_lookup[label]
		    orig_graph_map[-1][ind][label] = orig_graph_map[-1][ind].get(label, 0) + 1
		
		ind += 1
		
	compressed_labels = copy.deepcopy(labels)

	# WL iterations
	for it in range(h):
		unique_labels_per_h = set()
		label_lookup = {}
		ind = 0
		for G in graphs:
		    node2index = {}
		    for node in G.nodes():
		        node2index[node] = len(node2index)
		        
		    for node in G.nodes():
		        node_label = tuple([labels[ind][node2index[node]]])
		        neighbors = G.neighbors(node)
		        if len(neighbors) > 0:
		            neighbors_label = tuple([labels[ind][node2index[neigh]] for neigh in neighbors])
		            node_label =  str(node_label) + "-" + str(sorted(neighbors_label))
		        if not label_lookup.has_key(node_label):
		            label_lookup[node_label] = len(label_lookup)
		            
		        compressed_labels[ind][node2index[node]] = label_lookup[node_label]
		        orig_graph_map[it][ind][node_label] = orig_graph_map[it][ind].get(node_label, 0) + 1
		        
		    ind +=1
		    
		labels = copy.deepcopy(compressed_labels)
	
	K = np.zeros((N, N))
	for it in range(-1, h):
		for i in range(N):
			for j in range(N):
			    common_keys = set(orig_graph_map[it][i].keys()) & set(orig_graph_map[it][j].keys())
			    K[i][j] += sum([orig_graph_map[it][i].get(k,0)*orig_graph_map[it][j].get(k,0) for k in common_keys])
				  	                            
	return K


if __name__ == "__main__":
    # read the parameters
    ds_name = sys.argv[1]
    h = int(sys.argv[2])

    graphs, labels = load_data(ds_name)
    np.save(ds_name+"_labels", labels)
    
    print("Building wl subtree kernel for "+ds_name) 
        
    K = wl_subtree_kernel(graphs, h)
    np.save(ds_name+"_wl_subtree_"+str(h), K)