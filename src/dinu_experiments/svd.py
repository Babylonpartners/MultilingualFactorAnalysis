# Copyright 2019 Babylon Partners Limited. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import numpy as np
from .utils import normalize, make_training_matrices


def learn_transformation(source_matrix, target_matrix, normalize_vectors=True, svd_mean=False):
    """
    Source and target matrices are numpy arrays, shape
    (dictionary_length, embedding_dimension). These contain paired
    word vectors from the bilingual dictionary.
    """
    # optionally normalize the training vectors
    if normalize_vectors:
        source_matrix = normalize(source_matrix)
        target_matrix = normalize(target_matrix)

    # Optionally use MLE mean estimates to zero center before applying SVD
    if svd_mean:
        source_matrix = source_matrix - np.mean(source_matrix, axis=0)[None,:]
        target_matrix = target_matrix - np.mean(target_matrix, axis=0)[None,:]

    # perform the SVD
    product = np.matmul(source_matrix.transpose(), target_matrix)
    U, s, V = np.linalg.svd(product)

    # return orthogonal transformation which aligns source language to the target
    return np.matmul(U, V), np.mean(source_matrix, axis=0)[None,:], np.mean(target_matrix, axis=0)[None,:]


def svd_align(lang1_dictionary, lang2_dictionary, bilingual_dictionary):

    # form the training matrices
    source_matrix, target_matrix = make_training_matrices(
        lang1_dictionary, lang2_dictionary, bilingual_dictionary)

    # learn and apply the transformation
    transform = learn_transformation(source_matrix, target_matrix)
    lang1_dictionary.apply_transform(transform)

    return lang1_dictionary
