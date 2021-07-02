import os
from data.base_dataset import BaseDataset, get_transform, GlobalTransParams
from data.image_folder import make_dataset
from PIL import Image
import pandas as pd
import random
import math
from data.complex_data_processing.strtmask import *
from data.complex_data_processing.integral_grad import *


class StrtdemDerivativeAndIntegralDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)

        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.crop_pos_max = opt.load_size - opt.crop_size
        self.dd_loss = opt.dd_loss
        self.avghi_loss = opt.avghi_loss
        self.water_area_loss = opt.water_area_loss
        self.add_inner_random = opt.add_inner_random
        self.average_height = opt.average_height
        self.gen_height_diff = opt.gen_height_diff
        self.hidiff_output_nc = opt.hidiff_output_nc

        if self.gen_height_diff:
            self.dir_index = os.path.join(opt.dataroot, opt.phase + 'Index')  # the index file '/path/to/data/trainIndex'
            self.index_name = os.path.join(self.dir_index, 'index.csv')
            self.df = pd.read_csv(self.index_name)

        strt_mask_generator = StrtMaskGenerator(self.dir_B)
        if opt.cal_road_perpendicular_line:
            strt_mask_generator.need_gen_road_perpendicular_line(True)
        else:
            strt_mask_generator.need_gen_road_perpendicular_line(False)
        if opt.cal_water_area_mask:
            strt_mask_generator.need_water_area_mask(True)
        else:
            strt_mask_generator.need_water_area_mask(False)
        strt_mask_generator.work()

        if self.dd_loss:
            self.dir_rpl = strt_mask_generator.get_road_perpendicular_line_path()
            self.rpl_paths = sorted(make_dataset(self.dir_rpl, opt.max_dataset_size))  # load images from road perpendicular line path
            self.rpl_size = len(self.rpl_paths)  # get the size of dataset rpl

        if self.water_area_loss:
            self.dir_wam = strt_mask_generator.get_water_area_mask_path()
            self.wam_paths = sorted(make_dataset(self.dir_wam, opt.max_dataset_size))  # load images from water area mask path
            self.wam_size = len(self.wam_paths)  # get the size of dataset wam

        if self.add_inner_random:
            self.base_random = torch.rand((opt.inner_random_nc, 1, 1))

        btoA = self.opt.direction == 'BtoA'
        input_nc = self.opt.output_nc if btoA else self.opt.input_nc       # get the number of channels of input image
        output_nc = self.opt.input_nc if btoA else self.opt.output_nc      # get the number of channels of output image

        self.normal_bg_color = opt.background_color / 255.0

        params = {'crop_global': True, 'flip_global': True}
        self.strt_normal_params = {'need_norm': True, 'mean': (0.5, 0.5, 0.5),
                              'std': (0.5, 0.5, 0.5)}
        self.dem_normal_params = {'need_norm': True, 'mean': 0.5, 'std': 0.5}
        self.transform_A = get_transform(self.opt, params=params, grayscale=(input_nc == 1), normal_params=self.strt_normal_params)
        self.transform_B = get_transform(self.opt, params=params, grayscale=True, normal_params=self.dem_normal_params)
        self.transform_A_mask = get_transform(self.opt, params=params, grayscale=True,
                                              normal_params={'need_norm': False},
                                              custom_tensor_fun=self.preprocessing)
        self.transform_A_rpl = get_transform(self.opt, params=params, method=Image.NEAREST, normal_params={'need_norm': False})
        self.transform_A_wam = get_transform(self.opt, params=params, normal_params={'need_norm': False})

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
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
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

        B_grad = IntegralGrad.get_gradien(B)
        print('min: ' + str(torch.min(B_grad)))
        print('max: ' + str(torch.max(B_grad)))

        ret = {'A': A, 'B': B, 'B_grad': B_grad, 'A_paths': A_path, 'B_paths': B_path}

        A_mask = self.transform_A_mask(A_img)
        ret['A_mask'] = A_mask

        if self.dd_loss:
            rpl_path = self.rpl_paths[index % self.rpl_size]
            rpl_img = Image.open(rpl_path).convert('RGB')
            rpl = self.transform_A_rpl(rpl_img)
            ret['rpl'] = rpl

        if self.water_area_loss:
            wam_path = self.wam_paths[index % self.wam_size]
            wam_img = Image.open(wam_path).convert('RGB')
            wam = self.transform_A_wam(wam_img)
            ret['wam_area'] = wam[0:1, :, :]
            ret['wam_edge'] = wam[1:2, :, :]

        if self.add_inner_random:
            inner_random = self.base_random * index
            inner_random -= inner_random.int()
            ret['inner_random'] = inner_random

        if self.avghi_loss:
            avghi = (self.average_height / 255.0 - 0.5) / 0.5
            avg_height = torch.ones([1, 1, 1]) * avghi if self.average_height > 0 else torch.ones([1, 1, 1]) * torch.mean(B)
            ret['avg_height'] = avg_height

        if self.gen_height_diff:
            B_name = os.path.basename(B_path)
            row = self.df.loc[self.df['StreetMaps'] == B_name]
            hidiff = row.iat[0, 3] - row.iat[0, 2]
            hidiff_idx = math.ceil(hidiff / 255.0)
            if hidiff_idx == 0:
                hidiff_idx = 1  # hidiff_idx >= 1
            ret['hidiff_idx'] = hidiff_idx

        return ret

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)

    def get_preinput(self):
        preinput = {'A_norm': self.strt_normal_params, 'B_norm': self.dem_normal_params}
        return preinput

    def preprocessing(self, data_tensor):
        background = self.normal_bg_color
        data_tensor[data_tensor == background] = 0
        data_tensor[(data_tensor != background) & (data_tensor != 0)] = 1
        return data_tensor
