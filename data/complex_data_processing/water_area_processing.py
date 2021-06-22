import cv2 as cv
from data.complex_data_processing.map_elem_split import *


def draw_water_area_mask(cv_img):
    water_img = MapElemSplit.water_only(cv_img)
    water_ch_img = water_img.copy()
    water_ch_img[:, :, 1:] = 0

    water_img = cv.cvtColor(water_img, cv.COLOR_RGB2GRAY)

    kernel = cv.getStructuringElement(cv.MORPH_RECT, (7, 7))
    water_edge_img = cv.morphologyEx(water_img, cv.MORPH_DILATE, kernel)
    water_edge_img = water_edge_img - water_img

    water_ch_img[:, :, 1] = water_edge_img

    return water_ch_img
