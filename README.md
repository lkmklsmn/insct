# insct ("Insight")
**IN**tegration of millions of **S**ingle **C**cells using batch-aware **T**riplet networks

tnn calculates an integrated embedding for scRNA-seq data. With insct, you can:

* Integrate scRNA-seq datasets across batches with/without labels.
* Build a low-dimensional representation of the scRNA-seq data.
* Accurately predict cell types for an independent scRNAseq dataset.
* Integration of millions of cells on personal computers.

![tnn](https://github.com/lkmklsmn/insct/blob/master/TNN_schematic.jpg)

## Check out our interactive tutorials!
The following notebooks can be run within your web browser and allow you to interactively explore tnn. We have prepared the following analysis examples:
1. [Simulation dataset](https://github.com/lkmklsmn/bbtnn/tree/master/examples/TNN_Simulation.ipynb)
2. [Pancreas dataset](https://github.com/lkmklsmn/bbtnn/tree/master/examples/TNN_pancreas_comparison.ipynb)

## Installation

To install **insct**, follow these instructions:


### Github

Download the package from Github and install it locally:

```alias
git clone http://github.com/lkmklsmn/insct
cd insct
pip install .
```

## Input
### Unsupervised model
1. AnnData object with PCs
2. Batch vector

### Supervised model
Triplets sampled based on known labels
1. AnnData object with PCs
2. Batch vector
3. Celltype vector

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
