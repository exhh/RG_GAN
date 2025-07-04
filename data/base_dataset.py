"""This module implements an abstract base class (ABC) 'BaseDataset' for datasets.

It also includes common transformation functions (e.g., get_transform, __scale_width), which can be later used in subclasses.
"""
import random
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
from abc import ABC, abstractmethod


class BaseDataset(data.Dataset, ABC):


    def __init__(self, opt):
        self.opt = opt
        self.root = opt.dataroot

    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    @abstractmethod
    def __len__(self):
        return 0

    @abstractmethod
    def __getitem__(self, index):
        pass

def get_params_s1(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        new_h = new_w = opt.load_size
        
    elif opt.preprocess == 'scale_width_and_crop':
        new_w = opt.load_size
        new_h = opt.load_size * h // w
    
    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = random.random() < 0.5
    colorjitter = False
    return {'crop_pos': (x, y), 'flip': flip, 'colorjitter': colorjitter, 'new_h': new_h, 'new_w': new_w} #



def get_params(opt, size):
    w, h = size
    new_h = h
    new_w = w
    if opt.preprocess == 'resize_and_crop':
        if random.random() < 0.5: # random scaling
            scaling = np.random.uniform(0.9,1.1)
            new_h = new_w = int(opt.load_size * scaling)
        else:
            new_h = new_w = opt.load_size

    if random.random() < 0.5: #sheering
        sheering = np.random.uniform(0.9,1.1)#(0.8,1.2)
        new_w = int(opt.load_size * sheering)

    x = random.randint(0, np.maximum(0, new_w - opt.crop_size))
    y = random.randint(0, np.maximum(0, new_h - opt.crop_size))

    flip = False
    colorjitter = random.random() < 0.2

    return {'crop_pos': (x, y), 'flip': flip, 'colorjitter': colorjitter, 'new_h': new_h, 'new_w': new_w} #


def get_transform(opt, params=None, grayscale=False, method=Image.BICUBIC, convert=True, normalize=True):
    transform_list = []
    if grayscale:
        transform_list.append(transforms.Grayscale(1))
    if 'resize' in opt.preprocess:
        if params is None:
            osize = [opt.load_size, opt.load_size]
        else:
            osize = [params['new_w'], params['new_h']]
        transform_list.append(transforms.Resize(osize, method))

    if 'crop' in opt.preprocess:
        if params is None:
            transform_list.append(transforms.RandomCrop(opt.crop_size))
        else:
            transform_list.append(transforms.Lambda(lambda img: __crop(img, params['crop_pos'], opt.crop_size)))

    if opt.preprocess == 'none':
        transform_list.append(transforms.Lambda(lambda img: __make_power_2(img, base=4, method=method)))

    if not opt.no_flip:
        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: __flip(img, params['flip'])))
    
    if params is not None:
        if params['colorjitter']:
            cj = random.random()
            if cj < 0.4:
                transform_list.append(transforms.ColorJitter(brightness=(0.7,1.), contrast=(0.7,1.)))
            elif cj < 0.65:
                transform_list.append(transforms.ColorJitter(brightness=(1.,1.3), contrast=(0.7,1.)))
            elif cj < 0.9:
                transform_list.append(transforms.ColorJitter(brightness=(0.7,1.), contrast=(1.,1.3)))
            else:
                transform_list.append(transforms.ColorJitter(brightness=(1.,1.3), contrast=(1.,1.3)))
            

    if convert:
        transform_list += [transforms.ToTensor()]
        if normalize:
            if grayscale:
                transform_list += [transforms.Normalize((0.5,), (0.5,))]
            else:
                transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)


def __make_power_2(img, base, method=Image.BICUBIC):
    ow, oh = img.size
    h = int(round(oh / base) * base)
    w = int(round(ow / base) * base)
    if h == oh and w == ow:
        return img

    __print_size_warning(ow, oh, w, h)
    return img.resize((w, h), method)


def __crop(img, pos, size):
    ow, oh = img.size
    x1, y1 = pos
    tw = th = size
    if (ow > tw or oh > th):
        return img.crop((x1, y1, x1 + tw, y1 + th))
    return img


def __flip(img, flip):
    if flip:
        return img.transpose(Image.FLIP_LEFT_RIGHT)
    return img


def __print_size_warning(ow, oh, w, h):
    if not hasattr(__print_size_warning, 'has_printed'):
        print("The image size needs to be a multiple of 4. "
              "The loaded image size was (%d, %d), so it was adjusted to "
              "(%d, %d). This adjustment will be done to all images "
              "whose sizes are not multiples of 4" % (ow, oh, w, h))
        __print_size_warning.has_printed = True
