import os
from PIL import Image
import numpy as np
from data.complex_data_processing.road_processing import *
from data.complex_data_processing.water_area_processing import *


class StrtMaskGenerator:
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.root_path = os.path.join(self.dataset_path, '../Tmp')
        self.road_perpendicular_line_path = os.path.join(self.root_path, 'road_perpendicular_line')
        self.water_area_mask_path = os.path.join(self.root_path, 'water_area_mask')

        self.create_road_perpendicular_line = False
        self.create_water_area_mask = False
        return

    def need_gen_road_perpendicular_line(self, need_gen):
        self.check_road_perpendicular_line()
        self.create_road_perpendicular_line = need_gen
        return

    def check_road_perpendicular_line(self):
        need_gen = False
        if not os.path.exists(self.root_path):
            need_gen = True
            os.mkdir(self.root_path)
        if not os.path.exists(self.road_perpendicular_line_path):
            need_gen = True
            os.mkdir(self.road_perpendicular_line_path)
        return need_gen

    def gen_road_perpendicular_line(self):
        img_names = os.listdir(self.dataset_path)
        for img_name in img_names:
            img_path = os.path.join(self.dataset_path, img_name)
            A_img = Image.open(img_path).convert('RGB')
            cv_img = np.array(A_img)
            cv_img = draw_road_perpendicular_line(cv_img)
            cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
            cv.imwrite(os.path.join(self.road_perpendicular_line_path, img_name), cv_img)
            print(f'Finish {img_name}')
        return

    def need_water_area_mask(self, need_gen):
        self.check_water_area_mask()
        self.create_water_area_mask = need_gen
        return

    def check_water_area_mask(self):
        need_gen = False
        if not os.path.exists(self.root_path):
            need_gen = True
            os.mkdir(self.root_path)
        if not os.path.exists(self.water_area_mask_path):
            need_gen = True
            os.mkdir(self.water_area_mask_path)
        return need_gen

    def gen_water_area_mask(self):
        img_names = os.listdir(self.dataset_path)
        for img_name in img_names:
            img_path = os.path.join(self.dataset_path, img_name)
            A_img = Image.open(img_path).convert('RGB')
            cv_img = np.array(A_img)
            cv_img = draw_water_area_mask(cv_img)
            cv_img = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
            cv.imwrite(os.path.join(self.water_area_mask_path, img_name), cv_img)
        return

    def work(self):
        if self.create_road_perpendicular_line:
            self.gen_road_perpendicular_line()
        if self.create_water_area_mask:
            self.gen_water_area_mask()
        return

    def get_road_perpendicular_line_path(self):
        return self.road_perpendicular_line_path

    def get_water_area_mask_path(self):
        return self.water_area_mask_path
