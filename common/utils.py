import torch
import numpy as np
from sklearn import decomposition

def znorm(data):
    """
       Z-Normalization
    """
    mu = np.average(data, axis=0)
    std = np.std(data, axis=0)
    data = (data - mu) / std
    return data, mu, std


def pca_train(data, subspace_dim):
    '''
    :param data: data
    :param subspace_dim:  expected dimension of the subspace or the ratio of specturm be kept
    :return: trained pca model
    '''

    if isinstance(subspace_dim, float):
        var_ratio = np.linalg.svd(data - np.mean(data,0),full_matrices=False,compute_uv=False)**2

        cummulated = np.cumsum(var_ratio)
        for index in range(len(cummulated)):
            if cummulated[index] > subspace_dim:
                break
        subspace_dim = index

    pca = decomposition.PCA(n_components=subspace_dim)
    return pca.fit(data)


def pca_component(data, subspace_dim):
    '''
    :param data: data
    :param subspace_dim:  expected dimension of the subspace or the ratio of specturm be kept
    :return: pca component
    '''

    # compute variance percentage, if desired
    if isinstance(subspace_dim, float):
        pca = decomposition.PCA()
        pca.fit(data)
        var_ratio = pca.explained_variance_ratio_
        cummulated = np.cumsum(var_ratio)
        for index in range(len(cummulated)):
            if cummulated[index] > subspace_dim:
                break
        subspace_dim = index
        return pca.components_[:subspace_dim, ]
    else:
        pca = decomposition.PCA(subspace_dim)
        return pca.fit(data).components_

def kernel(ker, X, X2, gamma=None):
    '''
    :param ker: kernel type
    :param X:  data1
    :param X2: data2
    :param gamma: kernel paramater
    :return: the kernel between X and X2
    '''
    if not ker or ker == 'primal':
        return X
    elif ker == 'linear':
        if not X2:
            K = np.dot(X.T, X)
        else:
            K = np.dot(X.T, X2)
    elif ker == 'rbf':
        n1sq = np.sum(X ** 2, axis=0)
        n1 = X.shape[1]
        if not X2:
            D = (np.ones((n1, 1)) * n1sq).T + np.ones((n1, 1)) * n1sq - 2 * np.dot(X.T, X)
        else:
            n2sq = np.sum(X2 ** 2, axis=0)
            n2 = X2.shape[1]
            D = (np.ones((n2, 1)) * n1sq).T + np.ones((n1, 1)) * n2sq - 2 * np.dot(X.T, X)
        K = np.exp(-gamma * D)
    elif ker == 'sam':
        if not X2:
            D = np.dot(X.T, X)
        else:
            D = np.dot(X.T, X2)
        K = np.exp(-gamma * np.arccos(D) ** 2)
    return K

def euclidean_dist(x, y):
    '''
    :param x:  torch tensor of size N x D
    :param y: torch tensor of size M x D
    :return: euclidean between pair of x and y
    '''
    # y: M x D
    n = x.size(0)
    m = y.size(0)
    d = x.size(1)
    assert d == y.size(1)
    # n * 1 * d to n * m * d
    x = x.unsqueeze(1).expand(n, m, d)
    #1 * m * d to n * m * d
    y = y.unsqueeze(0).expand(n, m, d)

    return torch.pow(x - y, 2).sum(2)
