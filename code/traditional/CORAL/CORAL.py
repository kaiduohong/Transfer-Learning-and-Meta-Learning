# encoding=utf-8
"""
    Created on 16:31 2018/11/13 
    @author: Jindong Wang
"""

import os
import numpy as np
import scipy.io
import scipy.linalg
import sklearn.metrics
import sklearn.neighbors
from sklearn.svm import SVC
from sklearn import preprocessing
from common import setting


class CORAL:
    def __init__(self):
        super(CORAL, self).__init__()

    def fit(self, Xs, Xt):
        '''
        Perform CORAL on the source domain features
        :param Xs: ns * n_feature, source feature
        :param Xt: nt * n_feature, target feature
        :return: New source domain features
        '''
        _lambda = 1
        cov_src = np.cov(Xs.T) + _lambda * np.eye(Xs.shape[1])
        cov_tar = np.cov(Xt.T) + _lambda * np.eye(Xt.shape[1])
        A_coral = np.dot(np.real(scipy.linalg.fractional_matrix_power(cov_src, -0.5)),
                         np.real(scipy.linalg.fractional_matrix_power(cov_tar, 0.5)))
        Xs_new = np.dot(Xs, A_coral)
        return Xs_new

    def fit_predict(self, Xs, Ys, Xt, Yt):
        '''
        Perform CORAL, then predict using 1NN classifier
        :param Xs: ns * n_feature, source feature
        :param Ys: ns * 1, source label
        :param Xt: nt * n_feature, target feature
        :param Yt: nt * 1, target label
        :return: Accuracy and predicted labels of target domain
        '''

        Xs = preprocessing.scale(Xs)
        Xt = preprocessing.scale(Xt)

        Xs = preprocessing.normalize(Xs, norm='l2')
        Xt = preprocessing.normalize(Xt, norm='l2')


        Xs_new = self.fit(Xs, Xt)
        #clf = sklearn.neighbors.KNeighborsClassifier(n_neighbors=5)
        clf = SVC(C = 10, gamma = 1)
        clf.fit(Xs_new, Ys.ravel())
        y_pred = clf.predict(Xt)
        acc = sklearn.metrics.accuracy_score(Yt, y_pred)
        return acc, y_pred


if __name__ == '__main__':
    domains = ['caltech.mat', 'amazon.mat', 'webcam.mat', 'dslr.mat']

    data_root = setting.DATA_ROOT
    mat_root = os.path.join(data_root,'data_mat')



    for i in range(4):
        for j in range(4):
            if i != j:
                src, tar = os.path.join(mat_root,domains[i]), os.path.join(mat_root,domains[j])
                src_domain, tar_domain = scipy.io.loadmat(src), scipy.io.loadmat(tar)
                Xs, Ys, Xt, Yt = src_domain['feas'], src_domain['label'], tar_domain['feas'], tar_domain['label']

                coral = CORAL()
                acc, ypre = coral.fit_predict(Xs, Ys, Xt, Yt)
                print('source is {}, target is {}, acc = {}'.format(domains[i], domains[j], acc) )
