# insct ("Insight")
**IN**tegration of millions of **S**ingle **C**ells using batch-aware **T**riplet networks

insct calculates an integrated embedding for scRNA-seq data. With insct, you can:

* Integrate scRNA-seq datasets across batches with/without labels.
* Generate a low-dimensional representation of the scRNA-seq data.
* Integrate of millions of cells on personal computers.

## How does it work?
![tnn](https://github.com/lkmklsmn/insct/blob/master/TNN_schematic.jpg)

Overview of INSCT: a, INSCT learns a data representation, which integrates cells across batches. The goal of the network is to minimize the distance between Anchor and Positive while maximizing the distance between Anchor and Negative. Anchor and Positive pairs consist of transcriptionally similar cells from different batches. The Negative is a transcriptomically dissimilar cell sampled from the same batch as the Anchor. b, Principal components of three data points corresponding to Anchor, Positive and Negative are fed into three identical neural networks, which share weights. The triplet loss function is used to train the network weights and the two-dimensional embedding layer activations represent the integrated embedding.

![tnn](https://github.com/lkmklsmn/insct/blob/master/TNN_schematic.jpg)

## Check out our interactive tutorials!
The following notebooks can be run within your web browser and allow you to interactively explore tnn. We have prepared the following analysis examples:
1. [Simulation dataset](https://github.com/lkmklsmn/bbtnn/tree/master/examples/TNN_Simulation.ipynb)
2. [Pancreas dataset](https://github.com/lkmklsmn/bbtnn/tree/master/examples/TNN_pancreas_comparison.ipynb)

## Installation

To install **insct**, follow these instructions:


### Github

Install directly from Github using pip:

```alias
pip install git+https://github.com/lkmklsmn/insct.git
```

Download the package from Github and install it locally:

```alias
git clone http://github.com/lkmklsmn/insct
cd insct
pip install .
```

## Input
### Unsupervised model
Triplets sampled based on transcriptional similarity
1. AnnData object with PCs
2. Batch vector

### Supervised model
Triplets sampled based on known labels
1. AnnData object with PCs
2. Batch vector
3. Celltype vector

### Semi-supervised model
Triplets sampled based on both transcriptional similarity and known labels
1. AnnData object with PCs
2. Batch vector
3. Celltype vector
4. Masking vector (which labels to ignore)

## Output
1. Coordinates for the integrated embedding

## Usage
### Unsupervised model

```alias
from insct.tnn import TNN
model = TNN(k=50, distance='pn', batch_size=32, n_epochs_without_progress=10, k_to_m_ratio = 0.01)
model.fit(X = adata, Y = None, batch_name='batch')
```

### Supervised model
```alias
model = TNN(k=50, distance='pn', batch_size=64, n_epochs_without_progress=10, approx = False)
model.fit(X = adata, Y = None, batch_name='batch', celltype_name='Celltypes', cell_labeled = True)
```

### Semi-supervised model

```alias
model = TNN(k=50, distance='pn', batch_size=64, n_epochs_without_progress=10, approx = False)
model.fit(X = adata, Y = None, batch_name='batch', celltype_name='Celltypes', cell_labeled = True, mask_batch= batch_name)
```
