import os
from data.base_dataset import BaseDataset, get_transform, GlobalTransParams
from data.image_folder import make_dataset
from PIL import Image
import random
import torch
import cv2 as cv
import numpy as np


class PreprocessingDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt, ext_params=None):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        if ext_params is not None and 'use_extract_dataroot' in ext_params:
            self.dir_A = os.path.join(opt.extract_dataroot, opt.phase + 'A')
            self.dir_B = os.path.join(opt.extract_dataroot, opt.phase + 'B')
        else:
            self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
            self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))  # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))  # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc  # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc  # get the number of channels of output image
        self.need_A_mask = opt.mask_L1_loss
        self.random_background = opt.random_background
        self.net_branch_num = opt.net_branch_num
        self.ngf = opt.ngf
        self.img_size = opt.crop_size
        self.use_controlling_stick = opt.use_controlling_stick
        self.crop_pos_max = opt.load_size - opt.crop_size
        self.normal_bg_color = opt.background_color / 255.0
        self.fusion_controlling_stick = opt.fusion_controlling_stick

        self.mini_out = False
        if ext_params is not None:
            if 'mini_out' in ext_params:
                self.mini_out = ext_params['mini_out']
            else:
                self.mini_out = False

        self.controlling_sticks = [self.create_controlling_stick(controlling_stick_st=0),
                                   self.create_controlling_stick(controlling_stick_st=1),
                                   self.create_controlling_stick(controlling_stick_st=2),
                                   self.create_controlling_stick(controlling_stick_st=3),
                                   self.create_controlling_stick(controlling_stick_st=4)]

        params = {'crop_global': True, 'flip_global': True}
        self.transform_A = get_transform(self.opt, params=params, grayscale=(input_nc == 1))
        self.transform_B = get_transform(self.opt, params=params, grayscale=(output_nc == 1))
        self.transform_A_mask = get_transform(self.opt, params=params, grayscale=True,
                                              normal_params={'need_norm': False},
                                              custom_tensor_fun=self.preprocessing, )
        self.transform_B_mask = get_transform(self.opt, params=params, grayscale=True,
                                              normal_params={'need_norm': False},
                                              custom_tensor_fun=self.preprocessing, )

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:  # make sure index is within then range
            index_B = index % self.B_size
        else:  # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        # set random transform params
        GlobalTransParams.crop_pos = (random.randint(0, self.crop_pos_max),
                                      random.randint(0, self.crop_pos_max))
        GlobalTransParams.flip = True if random.random() > 0.5 else False
        # apply image transformation
        A = self.transform_A(A_img)
        B = self.transform_B(B_img)

        ret = {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

        B_mask = self.transform_B_mask(B_img)
        ret['B_mask'] = B_mask

        if self.need_A_mask:
            A_mask = self.transform_A_mask(A_img)
            ret['A_mask'] = A_mask

        if not self.mini_out:

            if self.random_background:
                # random_bg = torch.rand((1, self.img_size, self.img_size))
                random_bg = torch.zeros((1, self.img_size, self.img_size))
                ret['random_bg'] = random_bg

            if self.use_controlling_stick:
                ctrl_stick_idx = random.randint(0, len(self.controlling_sticks) - 1)
                ret['controlling_stick'] = self.controlling_sticks[ctrl_stick_idx]
                if self.fusion_controlling_stick:
                    fusion_ctrl_stick_idx = random.randint(0, len(self.controlling_sticks) - 2)
                    if fusion_ctrl_stick_idx == ctrl_stick_idx:
                        fusion_ctrl_stick_idx = len(self.controlling_sticks) - 1
                    ret['fusion_ctrlstick'] = self.controlling_sticks[fusion_ctrl_stick_idx]

        return ret

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def preprocessing(self, data_tensor):
        background = self.normal_bg_color
        data_tensor[data_tensor == background] = 0
        data_tensor[(data_tensor != background) & (data_tensor != 0)] = 1
        return data_tensor

    def create_controlling_stick(self, controlling_stick_st=1, controlling_stick_gap=5):
        controlling_stick = torch.zeros((1, self.img_size, self.img_size))
        for i in range(controlling_stick_st, self.img_size, controlling_stick_gap):
            controlling_stick[:, :, i] = 1
        return controlling_stick
