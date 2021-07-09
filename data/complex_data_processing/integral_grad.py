import os
from PIL import Image
import numpy as np
import cv2 as cv
import torch


class IntegralGrad:

    grad_norm_scale = 20

    def __init__(self):
        return

    @staticmethod
    def grad_merge(grad_x_mtx, grad_y_mtx, dim=-3):
        grad_mtx = torch.cat((grad_x_mtx, grad_y_mtx), dim=dim)
        return grad_mtx

    @staticmethod
    def grad_split(grad_mtx):
        grad_x_mtx = grad_mtx[..., 0:1, :, :]
        grad_y_mtx = grad_mtx[..., 1:2, :, :]
        return grad_x_mtx, grad_y_mtx

    @staticmethod
    def get_gradien_from_img(pic, grad_norm):
        ori_img = np.array(pic)
        if ori_img.ndim == 2:
            # if 2D image, add channel dimension (HWC)
            ori_img = np.expand_dims(ori_img, 2)
        pad_ori_img = np.pad(ori_img, ((0, 1), (0, 1), (0, 0)), 'edge')
        grad_mtx = IntegralGrad.get_gradien_img(pad_ori_img)
        grad_mtx = IntegralGrad.normalize(grad_mtx, grad_norm)
        grad_mtx = grad_mtx.permute(2, 0, 1)
        return grad_mtx

    @staticmethod
    def normalize(grad_mtx, grad_norm):
        if grad_norm['need_norm']:
            grad_mtx /= 20.0
            # grad_mtx /= (255.0 * 0.5)
            # grad_std = torch.tensor(grad_norm['std'])
            # grad_mean = torch.tensor(grad_norm['mean'])
            # grad_mtx = (grad_mtx - grad_mean) / grad_std
        return grad_mtx

    @staticmethod
    def get_gradien_img(ori_img):
        ori_mtx = torch.tensor(ori_img).float()
        grad_x_mtx = ori_mtx[:-1, 1:, :] - ori_mtx[:-1, :-1, :]
        grad_y_mtx = ori_mtx[1:, :-1, :] - ori_mtx[:-1, :-1, :]
        return IntegralGrad.grad_merge(grad_x_mtx, grad_y_mtx, dim=-1)

    @staticmethod
    def get_gradien(ori_mtx):
        ori_np = ori_mtx.numpy()
        if len(ori_np.shape) == 3:
            pad_mtd = ((0, 0), (0, 1), (0, 1))
        elif len(ori_np.shape) == 4:
            pad_mtd = ((0, 0), (0, 0), (0, 1), (0, 1))
        else:
            raise ValueError('Unknown ori_np.shape')
        ori_mtx = torch.tensor(np.pad(ori_np, pad_mtd, 'edge'))

        # [B, C, H, W]
        grad_x_mtx = ori_mtx[..., :, :-1, 1:] - ori_mtx[..., :, :-1, :-1]
        grad_y_mtx = ori_mtx[..., :, 1:, :-1] - ori_mtx[..., :, :-1, :-1]
        return IntegralGrad.grad_merge(grad_x_mtx, grad_y_mtx)

    @staticmethod
    def to_grad_norm(grad_mtx):
        grad_x_mtx, grad_y_mtx = IntegralGrad.grad_split(grad_mtx)
        grad_img_mtx = torch.sqrt(torch.pow(grad_x_mtx, 2) + torch.pow(grad_y_mtx, 2))
        grad_img_mtx = IntegralGrad.grad_norm_scale * grad_img_mtx
        return grad_img_mtx

    @staticmethod
    def to_grad_img(grad_mtx):
        grad_x_mtx, grad_y_mtx = IntegralGrad.grad_split(grad_mtx)
        grad_img_mtx = torch.sqrt(torch.pow(grad_x_mtx, 2) + torch.pow(grad_y_mtx, 2))
        grad_img = IntegralGrad.grad_norm_scale * grad_img_mtx.type(torch.uint8).numpy()
        return grad_img

    @staticmethod
    def integral_grad_path_x2y_auto_C(ori_mtx, grad_mtx, bottom=0):
        integrated_mtx = 0 * ori_mtx
        grad_x_mtx, grad_y_mtx = IntegralGrad.grad_split(grad_mtx)

        width = integrated_mtx.shape[-1]
        height = integrated_mtx.shape[-2]
        for x in range(0, width):
            for y in range(0, height):
                if x == 0 and y == 0:
                    integrated_mtx[..., 0, y, x] = bottom
                elif y == 0:
                    integrated_mtx[..., 0, y, x] = integrated_mtx[..., 0, y, x - 1] + grad_x_mtx[..., 0, y, x - 1]
                else:
                    integrated_mtx[..., 0, y, x] = integrated_mtx[..., 0, y - 1, x] + grad_y_mtx[..., 0, y - 1, x]

        min_val = torch.min(integrated_mtx)
        if min_val < bottom:
            integrated_mtx += (bottom - min_val)
        return integrated_mtx

    @staticmethod
    def integral_grad_path_y2x_auto_C(ori_mtx, grad_mtx, bottom=0):
        integrated_mtx = 0 * ori_mtx
        grad_x_mtx, grad_y_mtx = IntegralGrad.grad_split(grad_mtx)

        width = integrated_mtx.shape[-1]
        height = integrated_mtx.shape[-2]
        for y in range(0, height):
            for x in range(0, width):
                if x == 0 and y == 0:
                    integrated_mtx[..., 0, y, x] = bottom
                elif x == 0:
                    integrated_mtx[..., 0, y, x] = integrated_mtx[..., 0, y - 1, x] + grad_y_mtx[..., 0, y - 1, x]
                else:
                    integrated_mtx[..., 0, y, x] = integrated_mtx[..., 0, y, x - 1] + grad_x_mtx[..., 0, y, x - 1]

        min_val = torch.min(integrated_mtx)
        if min_val < bottom:
            integrated_mtx += (bottom - min_val)
        return integrated_mtx

    @staticmethod
    def integral_grad_mtx(ori_mtx, grad_mtx):
        integrated_mtx = 0 * ori_mtx
        grad_x_mtx, grad_y_mtx = IntegralGrad.grad_split(grad_mtx)

        width = integrated_mtx.shape[-1]
        height = integrated_mtx.shape[-2]
        for x in range(0, width):
            for y in range(0, height):
                if x == 0 and y == 0:
                    integrated_mtx[..., 0, y, x] = 0
                elif y == 0:
                    integrated_mtx[..., 0, y, x] = integrated_mtx[..., 0, y, x - 1] + grad_x_mtx[..., 0, y, x - 1]
                else:
                    integrated_mtx[..., 0, y, x] = integrated_mtx[..., 0, y - 1, x] + grad_y_mtx[..., 0, y - 1, x]
        return integrated_mtx

    @staticmethod
    def to_integrated_img(integrated_mtx):
        min_val = torch.min(integrated_mtx)
        if min_val < 0:
            integrated_mtx -= min_val
        return integrated_mtx.type(torch.uint8).numpy()
