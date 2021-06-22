import torch
import torch.nn as nn
import cv2 as cv


class DirectionalDerivativeLoss(nn.Module):
    def __init__(self) -> None:
        super(DirectionalDerivativeLoss, self).__init__()

    def forward(self, img, directional_deri_img):
        if img.shape[-3] != 1 or directional_deri_img.shape[-3] != 3:
            raise ValueError(
                f"The shape of input tensors are not correct, "
                f"got img channel {img.shape[-3]}, directional_deri_img channel {directional_deri_img.shape[-3]}."
            )
        # Sign correct
        # We use the third channel of image to express the sign of normalized_v
        # [abs(255 * normalized_v[0]), abs(255 * normalized_v[1]), v_sign]
        # For v_sign:
        # 0 = 00 --> normalized_v[0] and normalized_v[1] are all positive
        # 1 = 01 --> normalized_v[0] is negative and normalized_v[1] are positive
        # 2 = 10 --> normalized_v[0] is positive and normalized_v[1] are negative
        # 3 = 11 --> normalized_v[0] and normalized_v[1] are all negative
        sign_mtx = (255 * directional_deri_img[..., 2:, :-1, :-1]).int()  # directional_deri_img is in [0, 1]
        directional_deri_img[..., 0:1, :-1, :-1][sign_mtx & 1 == 1] *= -1
        directional_deri_img[..., 1:2, :-1, :-1][sign_mtx & (1 << 1) == (1 << 1)] *= -1

        directional_deri_img = directional_deri_img[..., :-1, :-1, :-1]
        pixel_diff = self.img_variation(img)
        dd_loss_mat = pixel_diff * directional_deri_img

        reduce_axes = (-3, -2, -1)
        res = dd_loss_mat.abs().sum(dim=reduce_axes)
        return res

    def img_variation(self, img):
        if len(img.shape) < 3 or len(img.shape) > 4:
            raise ValueError(
                f"Expected input tensor to be of ndim 3 or 4, but got {len(img.shape)}."
            )

        pixel_difh = img[..., 1:, :-1] - img[..., :-1, :-1]
        pixel_difw = img[..., :-1, 1:] - img[..., :-1, :-1]

        pixel_diff = torch.cat((pixel_difw, pixel_difh), -3)
        return pixel_diff
