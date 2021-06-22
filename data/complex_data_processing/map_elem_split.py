import torch
import numpy as np
import cv2 as cv
from data.complex_data_processing.colors import *


class MapElemSplit:
    bg_color = 244
    none_color = 0
    aim_color = 255
    mountain = Colors.mountain
    water = Colors.water
    neighborhood = Colors.neighborhood
    culture = Colors.culture
    vegetation = Colors.vegetation

    road_0 = Colors.road_0
    road_1 = Colors.road_1
    road_2 = Colors.road_2
    road_3 = Colors.road_3
    railway = Colors.railway

    def __init__(self):
        return

    @staticmethod
    def remove_bg(img):
        img[img == MapElemSplit.bg_color] = MapElemSplit.none_color
        img[(img != MapElemSplit.bg_color) & (img != MapElemSplit.none_color)] = MapElemSplit.aim_color
        return img

    @staticmethod
    def bg_only(img):
        img[img != MapElemSplit.bg_color] = MapElemSplit.none_color
        img[img == MapElemSplit.bg_color] = MapElemSplit.aim_color
        return img

    @staticmethod
    def road_only(img):
        height = img.shape[0]
        width = img.shape[1]
        if len(img.shape) == 3:
            channel = img.shape[2]
        else:
            return None

        ret_img = img * 0  # Reset to zero, but reserve the height and width
        black_delta = 10
        for h in range(0, height):
            for w in range(0, width):
                val = img[h, w]
                if all(val == MapElemSplit.road_0)\
                        or all(val == MapElemSplit.road_1) \
                        or all(val == MapElemSplit.road_2) \
                        or all(val == MapElemSplit.road_3) \
                        or (MapElemSplit.railway[0] - black_delta <= val[0] < MapElemSplit.railway[0] + black_delta
                            and MapElemSplit.railway[1] - black_delta <= val[1] < MapElemSplit.railway[1] + black_delta
                            and MapElemSplit.railway[2] - black_delta <= val[2] < MapElemSplit.railway[2] + black_delta):
                    ret_img[h, w] = MapElemSplit.aim_color
        return ret_img

    @staticmethod
    def remove_road(img):
        height = img.shape[0]
        width = img.shape[1]
        if len(img.shape) == 3:
            channel = img.shape[2]
        else:
            return None

        ret_img = img * 0 + MapElemSplit.none_color  # Reset to zero, but reserve the height and width
        for h in range(0, height):
            for w in range(0, width):
                val = img[h, w]
                if all(val == MapElemSplit.bg_color) \
                        or all(val == MapElemSplit.water) \
                        or all(val == MapElemSplit.neighborhood) \
                        or all(val == MapElemSplit.culture) \
                        or all(val == MapElemSplit.vegetation):
                    ret_img[h, w] = MapElemSplit.aim_color
        return ret_img

    @staticmethod
    def water_only(img):
        height = img.shape[0]
        width = img.shape[1]
        if len(img.shape) == 3:
            channel = img.shape[2]
        else:
            return None

        ret_img = img * 0 + MapElemSplit.none_color  # Reset to zero, but reserve the height and width
        for h in range(0, height):
            for w in range(0, width):
                val = img[h, w]
                if all(val == MapElemSplit.water):
                    ret_img[h, w] = MapElemSplit.aim_color
        return ret_img
