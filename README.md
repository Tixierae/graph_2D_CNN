# Graph classification with 2D CNNs ![GitHub stars](https://img.shields.io/github/stars/tixierae/graph_2D_CNN.svg?style=plastic) ![GitHub forks](https://img.shields.io/github/forks/tixierae/graph_2D_CNN.svg?color=blue&style=plastic)

[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-classification-with-2d-convolutional/graph-classification-on-collab)](https://paperswithcode.com/sota/graph-classification-on-collab?p=graph-classification-with-2d-convolutional) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-classification-with-2d-convolutional/graph-classification-on-re-m12k)](https://paperswithcode.com/sota/graph-classification-on-re-m12k?p=graph-classification-with-2d-convolutional) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-classification-with-2d-convolutional/graph-classification-on-re-m5k)](https://paperswithcode.com/sota/graph-classification-on-re-m5k?p=graph-classification-with-2d-convolutional) [![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/graph-classification-with-2d-convolutional/graph-classification-on-imdb-b)](https://paperswithcode.com/sota/graph-classification-on-imdb-b?p=graph-classification-with-2d-convolutional)

### What is this repo for?
This repo provides the code and datasets used in the paper [Classifying graphs as images with Convolutional Neural Networks](https://arxiv.org/abs/1708.02218) (Tixier, Nikolentzos, Meladianos and Vazirgiannis, 2017). Note that the paper was published at the ICANN 2019 conference under the title *Graph classification with 2D convolutional neural networks*. As its name suggests, the paper introduces a technique to perform graph classification with standard Convolutional Neural Networks for images (2D CNNs).

### Idea
We encode graphs as stacks of 2D histograms of their node embeddings, and pass them to a classical 2D CNN architecture designed for images. The *bins* of the histograms can be viewed as *pixels*, and the value of a given pixel is the number of nodes falling into the associated bin.

For instance, below are the node embeddings and corresponding bivariate histograms for graph ID #10001 (577 nodes, 1320 edges) of the REDDIT-12K dataset:
![alt text](https://github.com/Tixierae/graph_2D_CNN/raw/master/image_example_graph_cnn_github.png)
The full image representation of a graph is given by stacking its n_channels bivariate histograms (where n_channels can be 2,5...). Each pixel is thus associated with a n_channels-dimensional vector of counts.

### Results
Despite its simplicity, our method proves very competitive to state-of-the-art graph kernels, and even outperforms them by a wide margin on some datasets. 

10-fold CV average test set classification accuracy of state-of-the-art graph kernel and graph CNN baselines (top), vs our 2D CNN approach (bottom):
![alt text](https://github.com/Tixierae/graph_2D_CNN/raw/master/results_graph_cnn_github.png)

The results reported in the paper (without data augmentation) are available in the `/datasets/results/` subdirectory, with slight variations due to the stochasticity of the approach. You can read them using the `read_results.py` script.

### Advantages over graph kernels + SVM (GK+SVM)
We can summarize the advantages of our approach as follows:
* **better accuracy**: CNNs learn their own features directly from the raw data during training to optimize performance on the downstream task (whereas GKs compute similarity *a priori*)
* **better accuracy**: we compute images of graphs from their node embeddings (obtained via node2vec), so we capture both *local* and *global* information about the networks (whereas most GKs, based on substructures, capture only local information)
* **reduced time complexity at the graph level**: node2vec is linear in the number of nodes (whereas most GKs are polynomial) -> we can process bigger graphs
* **reduced time complexity at the collection level**: the time required to process a graph with a 2D CNN is constant (all images have same dimension for a given dataset), and the time required to go through the entire dataset with a 2D CNN grows linearly with the size of the dataset (whereas GKs take quadratic time to compute kernel matrix, then finding the support vectors is again quadratic) -> we can process bigger datasets


### Use
* `get_node2vec.py` computes the node2vec embeddings of the graphs from their adjacency matrices (parallelized over graphs)
* `get_histograms.py` computes the image representations of the graphs (stacks of 2D histograms) from their node2vec embeddings (parallelized over graphs)
* `main.py` reproduces the experiments in the paper (classification of graphs as images with a 2D CNN architecture, using a 10-fold cross validation scheme)
* `main_data_augmentation.py` is like `main.py`, but it implements the data augmentation scheme described in the paper (smoothed bootstrap)

Command line examples and descriptions of the parameters are available within each script.

### Setup 
Code was developed and tested under Ubuntu 16.04.2 LTS 64-bit operating system and Python 2.7 with [Keras 1.2.2](https://faroit.github.io/keras-docs/1.2.2/) and tensorflow 1.1.0 backend.

### Other notable dependencies
* igraph 0.7.1
* scikit-learn 0.18.1
* numpy 1.11.0
* multiprocessing
* functools
* json
* argparse

### Correspondence between names of datasets in the paper and in the code (paper -> code)
* IMDB-B -> imdb_action_romance
* COLLAB -> collab
* REDDIT-B -> reddit_iama_askreddit_atheism_trollx
* REDDIT-5K -> reddit_multi_5K
* REDDIT-12K -> reddit_subreddit_10K

### Cite
If you use some of the code in this repository in your work, please cite:

Conference version (ICANN 2019):
````BibTeX
@inproceedings{tixier2019graph,
  title={Graph classification with 2d convolutional neural networks},
  author={Tixier, Antoine J-P and Nikolentzos, Giannis and Meladianos, Polykarpos and Vazirgiannis, Michalis},
  booktitle={International Conference on Artificial Neural Networks},
  pages={578--593},
  year={2019},
  organization={Springer}
}
````

Pre-print version (2017):
````BibTeX
@article{tixier2017classifying,
  title={Classifying Graphs as Images with Convolutional Neural Networks},
  author={Tixier, Antoine Jean-Pierre and Nikolentzos, Giannis and Meladianos, Polykarpos and Vazirgiannis, Michalis},
  journal={arXiv preprint arXiv:1708.02218},
  year={2017}
}
````
