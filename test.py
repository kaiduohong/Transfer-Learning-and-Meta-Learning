#-*- coding:UTF-8 -*-
import numpy as np
import matplotlib.pyplot as plt

#import torch


P = np.matrix([[1,0.5,0,0],
                [0,0,0.5,0],
                [0,0.5,0,1],
                [0,0,0.5,0]]).T
a = np.matrix([[0],[1],[1],[1]])
S = np.matrix([[0],[5],[8],[9]])


print((P - np.eye(4)) * S)



plt.show()

#data_root = os.path.join(setting.DATE_ROOT,'data_mat')a


#cal_mat = loadmat(os.path.join(data_root,'caltech.mat'))

#print(cal_mat['label'].shape)