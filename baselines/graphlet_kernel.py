"""
- Sample code for:
"Classifying Graphs as Images with Convolutional Neural Networks"
arXiv preprint arXiv:1708.02218

- Computes the graphlet kernel by sampling "num_samples" graphlets of size 6 from each graph.

- Datasets should be placed into the "datasets" folder and can be downloaded from: https://ls11-www.cs.tu-dortmund.de/staff/morris/graphkerneldatasets

- To run, use the following command: 
python graphlet_kernel.py dataset num_samples
"""

import networkx as nx
import numpy as np
import sys
from math import factorial
from sympy.utilities.iterables import multiset_permutations

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


def generate_permutation_matrix():
	P = np.zeros((2**15,2**15),dtype=np.uint8)

	for a in range(2):
		for b in range(2):
			for c in range(2):
				for d in range(2):
					for e in range(2):
						for f in range(2):
							for g in range(2):
								for h in range(2):
									for i in range(2):
										for j in range(2):
											for k in range(2):
												for l in range(2):
													for m in range(2):
														for n in range(2):
															for o in range(2):	
																A = np.array([[0,a,b,c,d,e],[a,0,f,g,h,i],[b,f,0,j,k,l],[c,g,j,0,m,n],[d,h,k,m,0,o],[e,i,l,n,o,0]])

																perms = multiset_permutations(np.array(range(6),dtype=np.uint8))
																Per = np.zeros((factorial(6),6),dtype=np.uint8)
																ind = 0
																for permutation in perms:
																	Per[ind,:] = permutation
																	ind += 1

																for p in range(factorial(6)):
																	A_per = A[np.ix_(Per[p,:],Per[p,:])]
																	P[graphlet_type(A), graphlet_type(A_per)] = 1
	return P

											
def graphlet_type(A):
	factor = 2**np.array(range(15))
	
	upper = np.hstack((A[0,1:6],A[1,2:6],A[2,3:6],A[3,4:6],A[4,5]))
	result = np.sum(factor*upper)

	return int(result)


def graphlet_kernel(graphs, num_samples):
	N = len(graphs)

	Phi = np.zeros((N,2**15))

	P = generate_permutation_matrix()
  
	for i in range(len(graphs)):
	    n = graphs[i].number_of_nodes()
	    if n >= 6:           
			A = nx.to_numpy_matrix(graphs[i])
			A = np.asarray(A, dtype=np.uint8)
			for j in range(num_samples):
				r = np.random.permutation(n)
				window = A[np.ix_(r[:6],r[:6])]
				Phi[i, graphlet_type(window)] += 1

			Phi[i,:] /= num_samples

	K = np.dot(Phi,np.dot(P,np.transpose(Phi)))
	return K


if __name__ == "__main__":
    # read the parameters
    ds_name = sys.argv[1]
    num_samples = int(sys.argv[2])

    graphs, labels = load_data(ds_name)
    np.save(ds_name+"_labels", labels)
    
    print("Building graphlet kernel for "+ds_name) 
        
    K = graphlet_kernel(graphs, num_samples)
    np.save(ds_name+"_graphlet_"+str(num_samples)+"_samples", K)
