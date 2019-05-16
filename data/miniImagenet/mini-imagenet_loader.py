import os
import sys
import glob
from functools import partial
import numpy as np
from PIL import Image
import torch
from torchvision.transforms import ToTensor
from torchnet.dataset import ListDataset, TransformDataset
from torchnet.transform import compose

from common.data_utils import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler



def loader(opt):

    split_dir = os.path.join(opt.split_dir, opt.split_name)
    if opt.state == 'train':
        splits = opt.train_split_mode
    else:
        splits = ['test']

    ret = { }
    for split in splits:
        if split in ['val', 'test']:
            n_way = opt.test_way
            n_support = opt.test_shot
            n_query = opt.test_query
            n_episodes = opt.test_episodes
        else:
            n_way = opt.train_way
            n_support = opt.train_shot
            n_query = opt.train_query
            n_episodes = opt.train_episodes

