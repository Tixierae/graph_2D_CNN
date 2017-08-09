### What is this repo for?
This repo provides the code and data for the paper [Classifying graphs as images with Convolutional Neural Networks](https://arxiv.org/abs/1708.02218) (Tixier, Nikolentzos, Meladianos and Vazirgiannis, 2017)

### Idea
We encode graphs as stacks of 2D histograms of their node embeddings, and pass them to a classical 2D CNN architecture designed for images. Despite its simplicity, our method proves very competitive to state-of-the-art graph kernels, and even outperforms them by a wide margin on some datasets.

### Advantages over graph kernels + SVM (GK+SVM)
* **better accuracy**: CNNs learn their own features directly from the raw data during training to optimize performance on the downstream task (GKs compute similarity *a priori*)
* **better accuracy**: we compute images of graphs from their node embeddings (obtained via node2vec), so we capture both *local* and *global* information about the networks (most GKs, based on substructures, capture only local information)
* **reduced time complexity at the graph level**: node2vec is linear in the number of nodes (GKs are polynomial) -> we can process bigger graphs
* **reduced time complexity at the collection level**: the time required to process a graph with a 2D CNN is constant (all images have same dimension for a given dataset), and the time required to go through the entire dataset with a 2D CNN grows linearly with the size of the dataset (GKs take quadratic time to compute kernel matrix, then finding the support vectors is again quadratic) -> we can process bigger datasets

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