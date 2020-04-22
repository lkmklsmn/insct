import numpy as np
import scanpy as sc
import pandas as pd

from annoy import AnnoyIndex

from tensorflow import keras
from tensorflow.keras.models import load_model, Model
from tensorflow.keras.layers import Input, Dense, AlphaDropout, Lambda
from tensorflow.keras.regularizers import l2, l1_l2
from tensorflow.keras.callbacks import Callback, EarlyStopping
from tensorflow.keras import backend as K
from tensorflow.keras import regularizers
from tensorflow.keras.utils import Sequence

import random
from ivis.nn.losses import *
from ivis.nn.network import triplet_network, base_network
from ivis.nn.callbacks import ModelCheckpoint

from scipy.sparse import issparse
from scipy.spatial import cKDTree

from sklearn.base import BaseEstimator
from sklearn.neighbors import NearestNeighbors
import json
import os
import shutil
import multiprocessing
import platform

from sklearn.metrics.pairwise import rbf_kernel, euclidean_distances
from sklearn.neighbors import NearestNeighbors
from intervaltree import IntervalTree
import operator

from scipy.sparse import issparse
from annoy import AnnoyIndex
from multiprocessing import Process, cpu_count, Queue
from collections import namedtuple
from operator import attrgetter
from tqdm import tqdm
import time
import itertools
import networkx as nx

import hnswlib

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
from sklearn import metrics

def base_network(input_shape):
    '''Base network to be shared (eq. to feature extraction).
    '''
    inputs = Input(shape=input_shape)
    n_dim = round(0.75 * input_shape[0])
    x = Dense(n_dim, activation='selu',
              kernel_initializer='lecun_normal')(inputs)
    x = AlphaDropout(0.25)(x)
    x = Dense(n_dim, activation='selu',
              kernel_initializer='lecun_normal')(x)
    x = AlphaDropout(0.25)(x)
    x = Dense(n_dim, activation='selu', kernel_initializer='lecun_normal')(x)
    return Model(inputs, x)


def generator_from_index(adata, batch_name, celltype_name, mask_batch=None, cell_labeled=None, Y = None, k = 20, label_ratio = 0.8, k_to_m_ratio = 0.75, batch_size = 32, search_k=-1,
                         save_on_disk = True, approx = True, verbose=1):

    print('version 0.0.1. 15:00, 04/22/2020')

    # Calculate MNNs by pairwise comparison between batches
    
    cells = adata.obs_names
    
    if(verbose > 0):
        print("Calculating MNNs...")
           
    mnn_dict = create_dictionary_mnn(adata, batch_name=batch_name, k = k, save_on_disk = save_on_disk, approx = approx, verbose = verbose)
    
    if(verbose > 0):
        print(str(len(mnn_dict)) + " cells defined as MNNs")
        
    if cell_labeled == False:
        label_dict=dict()
    else:
        
        if (verbose > 0):
            print ('Generating supervised positive pairs...')

        label_dict_original = create_dictionary_label(adata, batch_name=batch_name, celltype_name=celltype_name, mask_batch=mask_batch,  k=k, verbose=verbose)
        num_label = round(label_ratio * len(label_dict_original))

        cells_for_label = np.random.choice(list(label_dict_original.keys()), num_label, replace = False)

        label_dict = {key: value for key, value in label_dict_original.items() if key in cells_for_label}

        if(verbose > 0):
            print(str(len(label_dict.keys())) + " cells defined as supervision triplets")

        print (len(set(mnn_dict.keys())&set(label_dict.keys())))

    if k_to_m_ratio == 0.0:
        knn_dict = dict()
        
    else:
        num_k = round(k_to_m_ratio * len(mnn_dict))
        # Calculate KNNs for subset of residual cells
        cells_for_knn = list(set(cells) - set(list(label_dict.keys()))| set(list(mnn_dict.keys())))
        if(len(cells_for_knn) > num_k):
            cells_for_knn = np.random.choice(cells_for_knn, num_k, replace = False)

        if(verbose > 0):
            print("Calculating KNNs...")

        cdata = adata[cells_for_knn]
        knn_dict = create_dictionary_knn(cdata, cells_for_knn, k = k, save_on_disk = save_on_disk, approx = approx)
        if(verbose > 0):
            print(str(len(cells_for_knn)) + " cells defined as KNNs")

    final_dict = merge_dict(mnn_dict, label_dict)
    final_dict.update(knn_dict)

    
    cells_for_train = list(final_dict.keys())
    print ('Total cells for training:'+ str(len(cells_for_train)))

    ddata = adata[cells_for_train]

    # Reorder triplet list according to cells
    if(verbose > 0):
        print("Reorder")
    names_as_dict = dict(zip(list(adata.obs_names), range(0, adata.shape[0])))
    def get_indices2(name):
          return([names_as_dict[x] for x in final_dict[name]])

    triplet_list = list(map(get_indices2, cells_for_train))
    
    batch_list = ddata.obs[batch_name]
    batch_indices = []
    for i in batch_list.unique():
        batch_indices.append(list(np.where(batch_list == i)[0]))

    batch_as_dict = dict(zip(list(batch_list.unique()), range(0, len(batch_list.unique()))))
    tmp = map(lambda _: batch_as_dict[_], batch_list)
    batch_list = list(tmp)

    if Y is None:
        return KnnTripletGenerator(X = ddata.obsm["X_pca"], X1 = adata.obsm['X_pca'], dictionary = triplet_list,
                               batch_list = batch_list, batch_indices = batch_indices, batch_size=batch_size)

    else:
        tmp = dict(zip(cells, Y))
        Y_new = [tmp[x] for x in cells_for_train]
        Y_new = le.fit_transform(Y_new)
        return LabeledKnnTripletGenerator(X = ddata.obsm["X_pca"], X1 = adata.obsm['X_pca'], Y = Y_new, dictionary = triplet_list,
                               batch_list = batch_list, batch_indices = batch_indices, batch_size = batch_size)


def merge_dict(x,y):
    for k,v in x.items():
                if k in y.keys():
                    y[k] += v
                else:
                    y[k] = v
    return y


class KnnTripletGenerator(Sequence):

    def __init__(self, X, X1, dictionary, batch_list, batch_indices, batch_size=32):
        self.X = X
        self.X1 = X1
        self.batch_list = batch_list
        self.batch_indices = batch_indices
        self.batch_size = batch_size
        self.dictionary = dictionary
        self.placeholder_labels = np.empty(batch_size, dtype=np.uint8)
        self.num_cells = len(self.dictionary)

    def __len__(self):
        return int(np.ceil(len(self.dictionary) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.num_cells))

        triplet_batch = [self.knn_triplet_from_dictionary(row_index = row_index,
                                                          neighbour_list = self.dictionary[row_index],
                                                          batch = self.batch_list[row_index],
                                                          num_cells = self.num_cells) for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)
        placeholder_labels = self.placeholder_labels[:triplet_batch.shape[0]]

        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), placeholder_labels

    def knn_triplet_from_dictionary(self, row_index, neighbour_list, batch, num_cells):
        """ A random (unweighted) positive example chosen. """
        triplets = []

        anchor = row_index
        positive = np.random.choice(neighbour_list)
        negative = self.batch_indices[batch][np.random.randint(len(self.batch_indices[batch]))]

        triplets += [self.X[anchor], self.X1[positive],
                     self.X1[negative]]

        return triplets


class LabeledKnnTripletGenerator(Sequence):
    def __init__(self, X, X1, Y, dictionary, batch_list, batch_indices, batch_size=32):
        self.X = X
        self.X1 = X1
        self.Y = Y
        self.batch_list = batch_list
        self.batch_indices = batch_indices
        self.batch_size = batch_size
        self.dictionary = dictionary
        self.num_cells = len(self.dictionary)

    def __len__(self):
        return int(np.ceil(len(self.dictionary) / float(self.batch_size)))

    def __getitem__(self, idx):

        batch_indices = range(idx * self.batch_size, min((idx + 1) * self.batch_size, self.num_cells))

        triplet_batch = [self.knn_triplet_from_dictionary(row_index = row_index,
                                                          neighbour_list = self.dictionary[row_index],
                                                          batch = self.batch_list[row_index],
                                                          num_cells = self.num_cells) for row_index in batch_indices]

        if (issparse(self.X)):
            triplet_batch = [[e.toarray()[0] for e in t] for t in triplet_batch]

        triplet_batch = np.array(triplet_batch)
        label_batch = self.Y[batch_indices]

        return tuple([triplet_batch[:, 0], triplet_batch[:, 1], triplet_batch[:, 2]]), tuple([np.array(label_batch), np.array(label_batch)])

    def knn_triplet_from_dictionary(self, row_index, neighbour_list, batch, num_cells):
        """ A random (unweighted) positive example chosen. """
        triplets = []

        anchor = row_index
        
        positive = np.random.choice(neighbour_list)
        negative = self.batch_indices[batch][np.random.randint(len(self.batch_indices[batch]))]

        triplets += [self.X[anchor], self.X1[positive],
                     self.X1[negative]]


        return triplets


def create_dictionary_label(bdata, batch_name, mask_batch, celltype_name, k=50, verbose=1):
    
    #cell_names = adata.obs_names
    adata = bdata[bdata.obs[batch_name]!=mask_batch]
    batch_list = adata.obs[batch_name]
    cell_types = adata.obs[celltype_name]

    print (batch_list.unique())
    
    types = []
    for i in batch_list.unique():
        types.append(cell_types[batch_list == i])

    print (len(types))
    
    labeled_dict = dict()

    for comb in list(itertools.permutations(range(len(types)), 2)):

        i = comb[0]
        j = comb[1]
    
        if(verbose > 0):
            print('Processing positive pairs {}'.format((i, j)))

        ref_types = types[i]
        new_types = types[j]
        common = set(ref_types) & set(new_types)

        for each in common:
            ref = list(ref_types[ref_types==each].index)
            new = list(new_types[new_types==each].index)
            
            num_k =min(int(k/10), 5,len(new))
            
            for key in ref:
                new_cells = np.random.choice(new, num_k, replace = False)
                if key not in labeled_dict.keys():
                    
                    labeled_dict[key] = list(new_cells)     
                else:
                    labeled_dict[key] += list(new_cells)
 
    return(labeled_dict)


def create_dictionary_mnn(adata, batch_name, k = 50, save_on_disk = True, approx = True, verbose = 1):

    cell_names = adata.obs_names

    batch_list = adata.obs[batch_name]
    datasets = []
    datasets_pcs = []
    cells = []
    for i in batch_list.unique():
        datasets.append(adata[batch_list == i])
        datasets_pcs.append(adata[batch_list == i].obsm["X_pca"])
        cells.append(cell_names[batch_list == i])

    mnns = dict()

    for comb in list(itertools.combinations(range(len(cells)), 2)):
        i = comb[0]
        j = comb[1]

        if(verbose > 0):
            print('Processing datasets {}'.format((i, j)))

        new = list(cells[j])
        ref = list(cells[i])

        ds1 = adata[new].obsm['X_pca']
        ds2 = adata[ref].obsm['X_pca']
        names1 = new
        names2 = ref
        match = mnn(ds1, ds2, names1, names2, knn=k, save_on_disk = save_on_disk, approx = approx)

        G = nx.Graph()
        G.add_edges_from(match)
        node_names = np.array(G.nodes)
        anchors = list(node_names)
        adj = nx.adjacency_matrix(G)

        tmp = np.split(adj.indices, adj.indptr[1:-1])

        for i in range(0, len(anchors)):
            key = anchors[i]
            i = tmp[i]
            names = list(node_names[i])
            mnns[key] = names

    return(mnns)


def create_dictionary_knn(adata, cell_subset, k = 50, save_on_disk = True, approx = True):

    cell_names = adata.obs_names

    dataset = adata[cell_subset]
    pcs = dataset.obsm['X_pca']

    def get_names(ind):
        return np.array(cell_subset)[ind]

    if approx:
        dim = pcs.shape[1]
        num_elements = pcs.shape[0]
        p = hnswlib.Index(space='l2', dim = dim)
        p.init_index(max_elements=num_elements, ef_construction=100, M=16)
        p.set_ef(10)
        p.add_items(pcs)
        ind, distances = p.knn_query(pcs, k=k)

        cell_subset = np.array(cell_subset)
        names = list(map(lambda x: cell_subset[x], ind))
        knns = dict(zip(cell_subset, names))

    else:
        nn_ = NearestNeighbors(n_neighbors = k, p = 2)
        nn_.fit(pcs)
        ind = nn_.kneighbors(pcs, return_distance=False)

        names = list(map(lambda x: cell_subset[x], ind))
        knns = dict(zip(cell_subset, names))

    return(knns)


class TNN(BaseEstimator):

    def __init__(self, embedding_dims=2, k=150, distance='pn', batch_size=64,
                 epochs=1000, n_epochs_without_progress=20,
                 margin=1, ntrees=50, search_k=-1,
                 precompute=True, save_on_disk=True,
                 k_to_m_ratio = 0.75,
                 label_ratio=0.75,
                 supervision_metric='sparse_categorical_crossentropy',
                 supervision_weight=0.3, annoy_index_path=None,
                 approx = True,
                 #sample_weight = None,
                 callbacks=[], build_index_on_disk=None, verbose=1):

        self.embedding_dims = embedding_dims
        self.k = k
        self.distance = distance
        self.batch_size = batch_size
        self.epochs = epochs
        self.n_epochs_without_progress = n_epochs_without_progress
        self.margin = margin
        self.ntrees = ntrees
        self.search_k = search_k
        self.precompute = precompute
        self.model_def = "dummy"
        self.model_ = None
        self.encoder = None
        self.k_to_m_ratio = k_to_m_ratio
        self.label_ratio = label_ratio
        self.approx = approx
        self.supervision_metric = supervision_metric
        self.supervision_weight = supervision_weight
        self.supervised_model_ = None
        self.loss_history_ = []
        self.annoy_index_path = annoy_index_path
        self.callbacks = callbacks
        #self.sample_weight = sample_weight
        self.save_on_disk = save_on_disk
        for callback in self.callbacks:
            if isinstance(callback, ModelCheckpoint):
                callback = callback.register_ivis_model(self)
        if build_index_on_disk is None:
            self.build_index_on_disk = True if platform.system() != 'Windows' else False
        else:
            self.build_index_on_disk = build_index_on_disk
        self.verbose = verbose

    def __getstate__(self):
        """ Return object serializable variable dict """

        state = dict(self.__dict__)
        if 'model_' in state:
            state['model_'] = None
        if 'encoder' in state:
            state['encoder'] = None
        if 'supervised_model_' in state:
            state['supervised_model_'] = None
        if 'callbacks' in state:
            state['callbacks'] = []
        if not isinstance(state['model_def'], str):
            state['model_def'] = None
        return state

    def _fit(self, X, batch_name, celltype_name=None, mask_batch=None, cell_labeled=None, Y=None, shuffle_mode=True):

        datagen = generator_from_index(X,
                                        batch_name = batch_name,
                                        celltype_name = celltype_name,
                                        mask_batch=mask_batch,
                                       cell_labeled = cell_labeled, 
                                        Y = Y,
                                        k_to_m_ratio = self.k_to_m_ratio,
                                       label_ratio = self.label_ratio,
                                       k=self.k,
                                       batch_size=self.batch_size,
                                       search_k=self.search_k,
                                       verbose = self.verbose,
                                       save_on_disk = self.save_on_disk,
                                       approx = self.approx)

        loss_monitor = 'loss'
        try:
            triplet_loss_func = triplet_loss(distance=self.distance,
                                             margin=self.margin)
        except KeyError:
            raise ValueError('Loss function `{}` not implemented.'.format(self.distance))

        if self.model_ is None:
            if type(self.model_def) is str:
                input_size = (X.obsm['X_pca'].shape[-1],)
                self.model_, anchor_embedding, _, _ = \
                    triplet_network(base_network(input_size),
                                    embedding_dims=self.embedding_dims)
            else:
                self.model_, anchor_embedding, _, _ = \
                    triplet_network(self.model_def,
                                    embedding_dims=self.embedding_dims)

            if Y is None:

                self.model_.compile(optimizer='adam', loss=triplet_loss_func)
            else:
                Y = le.fit_transform(Y)
                if is_categorical(self.supervision_metric):
                    if not is_multiclass(self.supervision_metric):
                        if not is_hinge(self.supervision_metric):
                            # Binary logistic classifier
                            if len(Y.shape) > 1:
                                self.n_classes = Y.shape[-1]
                            else:
                                self.n_classes = 1
                            supervised_output = Dense(self.n_classes, activation='sigmoid',
                                                      name='supervised')(anchor_embedding)
                        else:
                            # Binary Linear SVM output
                            if len(Y.shape) > 1:
                                self.n_classes = Y.shape[-1]
                            else:
                                self.n_classes = 1
                            supervised_output = Dense(self.n_classes, activation='linear',
                                                      name='supervised',
                                                      kernel_regularizer=regularizers.l1(l1=0.01))(anchor_embedding)
                    else:
                        if not is_hinge(self.supervision_metric):
                            validate_sparse_labels(Y)
                            self.n_classes = len(np.unique(Y[Y != np.array(-1)]))
                            # Softmax classifier
                            supervised_output = Dense(self.n_classes, activation='softmax',
                                                      name='supervised')(anchor_embedding)
                        else:
                            self.n_classes = len(np.unique(Y, axis=0))
                            # Multiclass Linear SVM output
                            supervised_output = Dense(self.n_classes, activation='linear',
                                                      name='supervised',
                                                      kernel_regularizer=regularizers.l1(l1=0.01))(anchor_embedding)
                else:
                    # Regression
                    if len(Y.shape) > 1:
                        self.n_classes = Y.shape[-1]
                    else:
                        self.n_classes = 1
                    supervised_output = Dense(self.n_classes, activation='linear',
                                              name='supervised')(anchor_embedding)

                supervised_loss = keras.losses.get(self.supervision_metric)
                if self.supervision_metric == 'sparse_categorical_crossentropy':
                    supervised_loss = semi_supervised_loss(supervised_loss)

                final_network = Model(inputs=self.model_.inputs,
                                      outputs=[self.model_.output,
                                               supervised_output])
                self.model_ = final_network
                self.model_.compile(
                    optimizer='adam',
                    loss={
                        'stacked_triplets': triplet_loss_func,
                        'supervised': supervised_loss
                         },
                    loss_weights={
                        'stacked_triplets': 1 - self.supervision_weight,
                        'supervised': self.supervision_weight})

                # Store dedicated classification model
                supervised_model_input = Input(shape=(X.obsm['X_pca'].shape[-1],))
                embedding = self.model_.layers[3](supervised_model_input)
                softmax_out = self.model_.layers[-1](embedding)

                self.supervised_model_ = Model(supervised_model_input, softmax_out)

        self.encoder = self.model_.layers[3]

        if self.verbose > 0:
            print('Training neural network')

        hist = self.model_.fit(
            datagen,
            epochs=self.epochs,
            callbacks=[callback for callback in self.callbacks] +
                      [EarlyStopping(monitor=loss_monitor,
                       patience=self.n_epochs_without_progress)],
            shuffle=shuffle_mode,
            workers = 10,
            verbose=self.verbose)

        self.loss_history_ += hist.history['loss']

    def fit(self, X, batch_name, celltype_name=None, mask_batch=None, cell_labeled=False,  Y=None, shuffle_mode=True):
        """Fit model.
        Parameters
        ----------
        X : Anndata object to be embedded.
        batch_name : name of column in Anndata.obs containing batch information
        Y : Optional array for supervised dimentionality reduction.
        Returns
        -------
        returns an instance of self
        """
        self._fit(X, batch_name, celltype_name, mask_batch, cell_labeled, Y, shuffle_mode = shuffle_mode)
        return self

    def fit_transform(self, X, batch_name, celltype_name=None, mask_batch=None, cell_labeled=None, Y=None, shuffle_mode=True):
        """Fit to data then transform
        Parameters
        ----------
        X : Anndata object to be embedded.
        Y : Optional array for supervised dimentionality reduction.
        Returns
        -------
        X_new : transformed array, shape (n_samples, embedding_dims)
            Embedding of the new data in low-dimensional space.
        """
        self.fit(X, batch_name, celltype_name,  mask_batch, Y, shuffle_mode)
        return self.transform(X)

    def transform(self, X):
        """Transform X into the existing embedded space and return that
        transformed output.
        Parameters
        ----------
        X : Anndata object to be embedded.
        Returns
        -------
        X_new : array, shape (n_samples, embedding_dims)
            Embedding of the new data in low-dimensional space.
        """
        embedding = self.encoder.predict(X.obsm['X_pca'], verbose=self.verbose)
        return embedding

    def score_samples(self, X):
        """Passes X through classification network to obtain predicted
        supervised values. Only applicable when trained in
        supervised mode.
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Data to be passed through classification network.
        Returns
        -------
        X_new : array, shape (n_samples, embedding_dims)
            Softmax class probabilities of the data.
        """
        if self.supervised_model_ is None:
            raise Exception("Model was not trained in classification mode.")

        softmax_output = self.supervised_model_.predict(X, verbose=self.verbose)
        return softmax_output


def semi_supervised_loss(loss_function):
    def new_loss_function(y_true, y_pred):
        mask = tf.cast(~tf.math.equal(y_true, -1), tf.float32)
        y_true_pos = tf.nn.relu(y_true)
        loss = loss_function(y_true_pos, y_pred)
        masked_loss = loss * mask
        return masked_loss
    new_func = new_loss_function
    new_func.__name__ = loss_function.__name__
    return new_func


def validate_sparse_labels(Y):
    if not zero_indexed(Y):
        raise ValueError('Ensure that your labels are zero-indexed')
    if not consecutive_indexed(Y):
        raise ValueError('Ensure that your labels are indexed consecutively')


def zero_indexed(Y):
    if min(abs(Y)) != 0:
        return False
    return True


def consecutive_indexed(Y):
    """ Assumes that Y is zero-indexed. """
    n_classes = len(np.unique(Y[Y != np.array(-1)]))
    if max(Y) >= n_classes:
        return False
    return True


def nn_approx(ds1, ds2, names1, names2, knn=50):
    dim = ds2.shape[1]
    num_elements = ds2.shape[0]
    p = hnswlib.Index(space='l2', dim=dim)
    p.init_index(max_elements=num_elements, ef_construction=100, M = 16)
    p.set_ef(10)
    p.add_items(ds2)
    ind,  distances = p.knn_query(ds1, k=knn)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def nn(ds1, ds2, names1, names2, knn=50, metric_p=2):
    # Find nearest neighbors of first dataset.
    nn_ = NearestNeighbors(knn, p=metric_p)
    nn_.fit(ds2)
    ind = nn_.kneighbors(ds1, return_distance=False)

    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def nn_annoy(ds1, ds2, names1, names2, knn = 20, metric='euclidean', n_trees = 50, save_on_disk = True):
    """ Assumes that Y is zero-indexed. """
    # Build index.
    a = AnnoyIndex(ds2.shape[1], metric=metric)
    if(save_on_disk):
        a.on_disk_build('annoy.index')
    for i in range(ds2.shape[0]):
        a.add_item(i, ds2[i, :])
    a.build(n_trees)

    # Search index.
    ind = []
    for i in range(ds1.shape[0]):
        ind.append(a.get_nns_by_vector(ds1[i, :], knn, search_k=-1))
    ind = np.array(ind)

    # Match.
    match = set()
    for a, b in zip(range(ds1.shape[0]), ind):
        for b_i in b:
            match.add((names1[a], names2[b_i]))

    return match


def mnn(ds1, ds2, names1, names2, knn = 20, save_on_disk = True, approx = True):
    # Find nearest neighbors in first direction.
    if approx:
        match1 = nn_approx(ds1, ds2, names1, names2, knn=knn)#, save_on_disk = save_on_disk)
        # Find nearest neighbors in second direction.
        match2 = nn_approx(ds2, ds1, names2, names1, knn=knn)#, save_on_disk = save_on_disk)
    else:
        match1 = nn(ds1, ds2, names1, names2, knn=knn)
        match2 = nn(ds2, ds1, names2, names1, knn=knn)
    # Compute mutual nearest neighbors.
    mutual = match1 & set([ (b, a) for a, b in match2 ])

    return mutual
