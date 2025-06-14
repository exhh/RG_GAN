import os
from data.base_dataset import BaseDataset, get_transform, get_params, get_params_s1
from data.image_folder import make_dataset
from PIL import Image
import random
import glob
import numpy as np
import skimage
from scipy.ndimage import gaussian_filter, binary_dilation, generate_binary_structure

class UnalignedDataset(BaseDataset):

    def __init__(self, opt):

        BaseDataset.__init__(self, opt)

        self.data_dir = 'lesion_detection/dotatate'
        self.realdata_dir = 'lesion_detection/datasets'
        self.base_data = 'based_folder'
        self.style_data = 'liver'

        self.A_paths = []
        train_base = ["%.3d" % i for i in np.array([0, 1, 2, 3, 4, 5, 9, 11, 12, 14, 15, 16, 18, 19, 20, 21, 22, 23, 26, 28, 29, \
                                               31, 33, 34, 35, 36, 38, 40, 44, 46, 47, 51, 53, 54, 55, 57, 58, 59, 60, 61, 62, 64, \
                                                   65, 67, 69, 70, 72, 74, 75, 76, 77, 79, 80, 82, 83, 85, 87, 91, 97, 98, 101, 104, 105])]
        train_base_arr = np.empty(len(train_base)).astype(str)
        for i in range(len(train_base)):
            train_base_arr[i] = r'DOTA011_' + train_base[i]
        train_base_data = tuple(train_base_arr)
        for id in train_base_data:
            self.A_paths += sorted(glob.glob(os.path.join(self.data_dir, self.base_data[1:], id, 'images/*.png')))

        del_list = []
        for idx, path in enumerate(self.A_paths):
            if int(path.split('/')[-1].split('.')[0]) > 80:
                del_list.append(idx)
        del_list.reverse()
        for del_idx in del_list:
            self.A_paths.pop(del_idx)

        self.B_paths = []

        # Need to specify the case IDs in the training set
        train_style_data = ('case_ID#10', 'case_ID#20')


        for id in train_style_data:
            self.B_paths += sorted(glob.glob(os.path.join(self.realdata_dir, self.style_data, id, 'images/*.png')))

        self.B_valid_paths = []

        # Need to specify the case IDs in the validation dataset
        valid_style_data = ('case_ID#1', 'case_ID#2')

        for id in valid_style_data:
            self.B_valid_paths += sorted(glob.glob(os.path.join(self.realdata_dir, self.style_data, id, 'images/*.png')))
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.B_valid_size = len(self.B_valid_paths)
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc
        if self.opt.epoch_count == 61: #designed augmentation for Stage 3
            params_A = get_params(self.opt, [128, 128])
            params_B = get_params(self.opt, [128, 128])
        else:
            params_A = get_params_s1(self.opt, [128, 128])
            params_B = get_params_s1(self.opt, [128, 128])
        self.transform_A = get_transform(self.opt, params=params_A, grayscale=(input_nc == 1), normalize=True)
        self.transform_A_labelmask = get_transform(self.opt, params=params_A, grayscale=(input_nc == 1), normalize=False)
        self.transform_B = get_transform(self.opt, params=params_B, grayscale=(output_nc == 1), normalize=True)
        self.transform_B_labelmask = get_transform(self.opt, params=params_B, grayscale=(output_nc == 1), normalize=False)

    def __getitem__(self, index):

        A_path = self.A_paths[index % self.A_size]
        if self.opt.serial_batches:
            index_B = index % self.B_size

        else:
            index_B = random.randint(0, self.B_size - 1)

        A_path_list = A_path.split('/')
        A_label_path = os.path.join('/',A_path_list[1], A_path_list[2], A_path_list[3], A_path_list[4], A_path_list[5], A_path_list[6], A_path_list[7], 'lesions', A_path_list[9])
        A_mask_path = os.path.join(self.data_dir, self.base_data[1:], 'DOTA011_liverMask', A_path_list[9])
        B_path = self.B_paths[index_B]

        A_img = Image.open(A_path).convert('L')
        A_label_PIL = Image.open(A_label_path).convert('L')
        A_mask_PIL = Image.open(A_mask_path).convert('L')
        B_img = Image.open(B_path).convert('L')

        # apply image transformation
        if self.opt.epoch_count == 61: #designed augmentation for Stage 3
            if random.random() < 0.5: #random noise
                A = skimage.util.random_noise(self.transform_A(A_img), mode='speckle')
                B = skimage.util.random_noise(self.transform_B(B_img), mode='speckle')
            else:
                A = self.transform_A(A_img)
                B = self.transform_B(B_img)
        else:
            A = self.transform_A(A_img)
            B = self.transform_B(B_img)

        A_label = (self.transform_A_labelmask(A_label_PIL) > 0.5).long()

        A_label_blur = gaussian_filter(np.expand_dims(binary_dilation(A_label[0], structure=generate_binary_structure(2, 2), iterations=1) * 1.,0), sigma=1) * self.opt.region_weight
        A_cycle_mask = A_label_blur + 1 - A_label_blur.sum() / self.opt.crop_size**2
        A_mask = (self.transform_A_labelmask(A_mask_PIL) > 0.5).long()


        return {'A': A, 'A_label': A_label, 'A_mask': A_mask, 'A_cycle_mask': A_cycle_mask, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        return max(self.A_size, self.B_size)
