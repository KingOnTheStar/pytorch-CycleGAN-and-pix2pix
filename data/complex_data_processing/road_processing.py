import torch
import numpy as np
import cv2 as cv
from data.complex_data_processing.map_elem_split import *
import math


def get_road_edge(cv_img):
    cv_img = MapElemSplit.remove_bg(cv_img)

    _, cv_img = cv.threshold(cv_img, 0, 255, cv.THRESH_OTSU)

    cv_img = cv.GaussianBlur(cv_img, (3, 3), 0)
    cv_img = cv.Canny(cv_img, 100, 200)
    return cv_img


def draw_road_perpendicular_line(cv_img):
    height = cv_img.shape[0]
    width = cv_img.shape[1]
    resize_img = cv.resize(cv_img, (2 * width, 2 * height), interpolation=cv.INTER_NEAREST)

    no_road_img = MapElemSplit.remove_road(resize_img)
    no_road_img = cv.cvtColor(no_road_img, cv.COLOR_RGB2GRAY)

    road_img = MapElemSplit.road_only(resize_img)
    road_img = cv.cvtColor(road_img, cv.COLOR_RGB2GRAY)
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    #road_img = cv.morphologyEx(road_img, cv.MORPH_OPEN, kernel)
    road_img = cv.morphologyEx(road_img, cv.MORPH_DILATE, kernel)

    road_img_from_bg = cv.bitwise_not(no_road_img)
    road_img = cv.bitwise_and(road_img, road_img_from_bg)
    road_img = get_road_edge(road_img)
    road_perpendicular_line = RoadPerpendicularLine(cv_img, road_img, no_road_img)
    return road_perpendicular_line.get_perpendicular_line()


class RoadPerpendicularLine:
    def __init__(self, ori_img, road_img, no_road_img):
        self.road_img = road_img
        self.no_road_img = no_road_img
        self.step = 2
        self.kernel_size = 7
        self.height = road_img.shape[0]
        self.width = road_img.shape[1]
        if len(road_img.shape) == 2:
            self.channel = 1
        else:
            self.channel = road_img.shape[2]
        if self.channel is not 1:
            raise ValueError('self.channel is not 1')

        self.ori_img = ori_img
        self.width_scale_rate = ori_img.shape[1] / self.width
        self.height_scale_rate = ori_img.shape[0] / self.height
        self.ori_size = True
        self.probe_len = 3
        return

    def get_perpendicular_line(self):
        if self.ori_size:
            ret_img = 0 * self.ori_img.copy()
        else:
            ret_img = self.road_img.copy()
        for h in range(0, self.height, self.step):
            for w in range(0, self.width, self.step):
                now_v = self.get_pixel(self.road_img, w, h)
                if now_v is None or now_v == 0:
                    continue
                k, b = self.local_k(w, h, kernel_size=self.kernel_size)
                if k is not None:
                    k = k if k != 0 else 1e-24
                    v = np.transpose(np.array([[1, -1/k]]))
                    normalized_v = v / np.linalg.norm(v)

                    probe_x = int(w + self.probe_len * normalized_v[0])
                    probe_y = int(h + self.probe_len * normalized_v[1])
                    valid, sign = self.is_in_road(probe_x, probe_y)
                    if valid:
                        for i in range(1, 15):
                            probe_x = int(w + sign * i * normalized_v[0])
                            probe_y = int(h + sign * i * normalized_v[1])
                            valid, sign = self.is_in_road(probe_x, probe_y)
                            if not valid or sign == -1:
                                break
                            if self.ori_size:
                                probe_x = int(self.width_scale_rate * probe_x)
                                probe_y = int(self.height_scale_rate * probe_y)
                                # We use the third channel of image to express the sign of normalized_v
                                # [abs(255 * normalized_v[0]), abs(255 * normalized_v[1]), v_sign]
                                # For v_sign:
                                # 0 = 00 --> normalized_v[0] and normalized_v[1] are all positive
                                # 1 = 01 --> normalized_v[0] is negative and normalized_v[1] are positive
                                # 2 = 10 --> normalized_v[0] is positive and normalized_v[1] are negative
                                # 3 = 11 --> normalized_v[0] and normalized_v[1] are all negative
                                v_sign = 0
                                if normalized_v[0] < 0:
                                    v_sign |= 1
                                if normalized_v[1] < 0:
                                    v_sign |= 1 << 1
                                self.set_pixel(ret_img, probe_x, probe_y,
                                               (abs(255 * normalized_v[0]), abs(255 * normalized_v[1]), v_sign))
                            else:
                                self.set_pixel(ret_img, probe_x, probe_y, 255)
        return ret_img

    def local_k(self, w, h, kernel_size=5):
        x0 = int(w - kernel_size * 0.5)
        y0 = int(h - kernel_size * 0.5)
        px_array = []
        py_array = []
        for xi in range(0, kernel_size):
            for yi in range(0, kernel_size):
                x = x0 + xi
                y = y0 + yi
                val = self.get_pixel(self.road_img, x, y, defaul_val=0)
                if val > 0:
                    px_array.append(x)
                    py_array.append(y)
        if len(px_array) >= 3:
            x = np.array(px_array)
            y = np.array(py_array)
            A = np.vstack([x, np.ones(len(x))]).T
            k, b = np.linalg.lstsq(A, y, rcond=None)[0]
            return k, b
        return None, None

    def is_in_road(self, x, y):
        val = self.get_pixel(self.no_road_img, x, y)
        if val is not None:
            if val > 0:
                return True, -1
            else:
                return True, 1
        else:
            return False, -1

    def valid_pos(self, w, h):
        return 0 <= w < self.width and 0 <= h < self.height

    def get_pixel(self, img, w, h, defaul_val=None):
        if not self.valid_pos(w, h):
            return defaul_val
        return img[h, w]

    def set_pixel(self, img, w, h, val):
        if not self.valid_pos(w, h):
            return False
        img[h, w] = val
        return True
