import collections
import os
import scipy.io as sio
import tensorflow as tf
import numpy as np
import numpy.matlib as npm
import random
import cv2
from tqdm import tqdm

from scipy.spatial import distance


def create_indices(labels):
    old = labels[0]
    indices = dict()
    indices[old] = 0
    for x in xrange(len(labels) - 1):
        new = labels[x + 1]
        if old != new:
            indices[new] = x + 1
        old = new
    return indices


def generate_triplet(_labels,_n_samples):
    # retrieve loaded patches and labels
    labels = _labels
    # group labels in order to have O(1) search
    count = collections.Counter(labels)
    # index the labels in order to have O(1) search
    indices = create_indices(labels)
    # range for the sampling
    labels_size = len(labels) - 1
    # triplets ids
    _index_1 = []
    _index_2 = []
    _index_3 = []
    # generate the triplets
    pbar = tqdm(xrange(_n_samples))

    for x in pbar:
        pbar.set_description('Generating triplets')
        idx = random.randint(0, labels_size)
        num_samples = count[labels[idx]]
        begin_positives = indices[labels[idx]]

        offset_a, offset_p = random.sample(xrange(num_samples), 2)
        while offset_a == offset_p:
            offset_a, offset_p = random.sample(xrange(num_samples), 2)
        idx_a = begin_positives + offset_a
        idx_p = begin_positives + offset_p
        _index_1.append(idx_a)
        _index_2.append(idx_p)
        idx_n = random.randint(0, labels_size)
        while labels[idx_n] == labels[idx_a] and \
                        labels[idx_n] == labels[idx_p]:
            idx_n = random.randint(0, labels_size)
        _index_3.append(idx_n)

    _index_1 = np.array(_index_1)
    _index_2 = np.array(_index_2)
    _index_3 = np.array(_index_3)

    temp_index = np.arange(_index_1.shape[0])

    np.random.shuffle(temp_index)
    _index_1 = _index_1[temp_index]
    _index_2 = _index_2[temp_index]
    _index_3 = _index_3[temp_index]

    return _index_1,_index_2,_index_3