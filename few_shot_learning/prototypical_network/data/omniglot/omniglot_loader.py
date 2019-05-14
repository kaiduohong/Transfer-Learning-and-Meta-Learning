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

import protonets
from common.data_utils import convert_dict, CudaTransform, EpisodicBatchSampler, SequentialBatchSampler


#OMNIGLOT_DATA_DIR = os.path.dirname(__file__)
OMNIGLOT_CACHE = { }

def load_image_path(key, out_field, d):
    d[out_field] = Image.open(d[key])
    return d

def convert_tensor(key, d):
    d[key] = 1.0 - torch.from_numpy(np.array(d[key], np.float32, copy=False)).transpose(0, 1).contiguous().view(1, d[key].size[0], d[key].size[1])
    return d

def rotate_image(key, rot, d):
    d[key] = d[key].rotate(rot)
    return d

def scale_image(key, height, width, d):
    d[key] = d[key].resize((height, width))
    return d

# d class包含了类别目录,大类是字母集，小类是字符
def load_class_images(im_dir,d):
    if d['class'] not in OMNIGLOT_CACHE:
        alphabet, character, rot = d['class'].split('/')
        image_dir = os.path.join(im_dir, 'data', alphabet, character)

        class_images = sorted(glob.glob(os.path.join(image_dir, '*.png')))
        if len(class_images) == 0:
            raise Exception("No images found for omniglot class {} at {}. Did you run download_omniglot.sh first?".format(d['class'], image_dir))

        image_ds = TransformDataset(ListDataset(class_images),
                                    compose([partial(convert_dict, 'file_name'),
                                             partial(load_image_path, 'file_name', 'data'),
                                             partial(rotate_image, 'data', float(rot[3:])),
                                             partial(scale_image, 'data', 28, 28),
                                             partial(convert_tensor, 'data')]))

        loader = torch.utils.data.DataLoader(image_ds, batch_size=len(image_ds), shuffle=False)

        for sample in loader:
            OMNIGLOT_CACHE[d['class']] = sample['data']
            break # only need one sample because batch size equal to dataset length

    return { 'class': d['class'], 'data': OMNIGLOT_CACHE[d['class']] }

def extract_episode(n_support, n_query, d):
    # data: N x C x H x W
    n_examples = d['data'].size(0)

    if n_query == -1:
        n_query = n_examples - n_support

    example_inds = torch.randperm(n_examples)[:(n_support+n_query)]
    support_inds = example_inds[:n_support]
    query_inds = example_inds[n_support:]

    xs = d['data'][support_inds]
    xq = d['data'][query_inds]

    return {
        'class': d['class'],
        'xs': xs,
        'xq': xq
    }

def loader(opt, splits):

    split_dir = os.path.join(opt.data_path, 'splits', opt.split)

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

        transforms = [partial(convert_dict, 'class'),
                      partial(load_class_images, opt.data_path),
                      partial(extract_episode, n_support, n_query)]

        if opt.cuda:
            transforms.append(CudaTransform())

        transforms = compose(transforms)
        class_names = []

        with open(os.path.join(split_dir, "{:s}.txt".format(split)), 'r') as f:
            for class_name in f.readlines():
                class_names.append(class_name.rstrip('\n'))
        ds = TransformDataset(ListDataset(class_names), transforms)

        if opt.sequential:
            sampler = SequentialBatchSampler(len(ds))
        else:
            sampler = EpisodicBatchSampler(len(ds), n_way, n_episodes)

        # use num_workers=0, otherwise may receive duplicate episodes
        ret[split] = torch.utils.data.DataLoader(ds, batch_sampler=sampler, num_workers=0)

    return ret
