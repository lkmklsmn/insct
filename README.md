# insct ("Insight")
**IN**tegration of millions of **S**ingle **C**ells using batch-aware **T**riplet networks

insct is a deep learning algorithm which calculates an integrated embedding for scRNA-seq data. With insct, you can:

* Integrate scRNA-seq datasets across batches with/without labels.
* Generate a low-dimensional representation of the scRNA-seq data.
* Integrate of millions of cells on personal computers.

For more info check out our [preprint](https://www.biorxiv.org/content/10.1101/2020.05.16.100024v1).

## How does it work?
![tnn](https://github.com/lkmklsmn/insct/blob/master/TNN_schematic.jpg)

**a**, INSCT learns a data representation, which integrates cells across batches. The goal of the network is to minimize the distance between Anchor and Positive while maximizing the distance between Anchor and Negative. Anchor and Positive pairs consist of transcriptionally similar cells from different batches. The Negative is a transcriptomically dissimilar cell sampled from the same batch as the Anchor. **b**, Principal components of three data points corresponding to Anchor, Positive and Negative are fed into three identical neural networks, which share weights. The triplet loss function is used to train the network weights and the two-dimensional embedding layer activations represent the integrated embedding.

To learn an integrated embedding that overcomes batch effects, INSCT samples triplets in a batch-aware manner:

![tnn](https://github.com/lkmklsmn/insct/blob/master/Sampling_animation.gif)

## What does it do?

For example, we simulated scRNAseq data, where batch effects dominate the embedding:

![tnn](https://github.com/lkmklsmn/insct/blob/master/umap_embedding.png)

However, INSCT learns an integrated embedding where cells cluster by group instead of batch:

![tnn](https://github.com/lkmklsmn/insct/blob/master/Simulation_animation.gif)

## Check out our interactive tutorials!
The following notebooks can be run within your web browser and allow you to interactively explore tnn. We have prepared the following analysis examples:
1. [Simulation dataset](https://colab.research.google.com/drive/1LEDnRwFH2v166T-pUaCYb6TZMgfViO-W?usp=sharing)
2. [Pancreas dataset](https://colab.research.google.com/drive/1v_B0pXVYMqHsV2polaoRHkxflrNcQGej?usp=sharing)

Notebooks to reproduce the analyses described in our preprint can be found in the _reproducibility_ folder.

## Installation

**insct** depends on the following Python packages. These need to be installed separately:
```
ivis==1.7.2
scanpy
hnswlib
```

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
Triplets sampled based on both transcriptional similarity and known labels
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
model = TNN()
model.fit(X = adata, batch_name='batch')
```

### Supervised model
```alias
model = TNN()
model.fit(X = adata, batch_name='batch', celltype_name='Celltypes')
```

### Semi-supervised model

```alias
model = TNN()
model.fit(X = adata, batch_name='batch', celltype_name='Celltypes', mask_batch= batch_name)
```
