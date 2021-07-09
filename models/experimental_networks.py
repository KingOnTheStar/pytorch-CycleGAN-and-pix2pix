import random
import torch
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
from models.base_networks import *


class UnetAttentionMaskGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, net_branch_num=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetAttentionMaskGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=ngf * net_branch_num, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost UNet layer
        self.model = MultiCNNMaskBlock(output_nc, ngf, net_branch_num, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input, input_mask):
        """Standard forward"""
        return self.model(input, input_mask)


class UnetRandomAndMaskGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, net_branch_num=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetRandomAndMaskGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=net_branch_num, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost UNet layer
        self.model = MultiCNNMaskRandomBGBlock(output_nc, ngf, net_branch_num, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input, input_mask, input_random_bg):
        """Standard forward"""
        return self.model(input, input_mask, input_random_bg)


class PostMaskUnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, net_branch_num=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(PostMaskUnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(output_nc, ngf, input_nc=net_branch_num, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost UNet layer
        self.model = DisperseBlock(output_nc, ngf, net_branch_num, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


class MaskCollectionGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, net_branch_num=3, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(MaskCollectionGenerator, self).__init__()

        self.model = MaskCollectionBlock(output_nc, ngf, net_branch_num, input_nc=input_nc, submodule=None, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input, input_mask, input_random_bg):
        """Standard forward"""
        return self.model(input, input_mask, input_random_bg)


class UnetInnerRandomGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, inner_ap_nc=0, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            inner_ap_nc -- the number of channels in inner append vector
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetInnerRandomGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionInnerRandomBlock(ngf * 8, ngf * 8, inner_ap_nc, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionInnerRandomBlock(ngf * 8, ngf * 8, inner_ap_nc, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionInnerRandomBlock(ngf * 4, ngf * 8, inner_ap_nc, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionInnerRandomBlock(ngf * 2, ngf * 4, inner_ap_nc, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionInnerRandomBlock(ngf, ngf * 2, inner_ap_nc, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionInnerRandomBlock(output_nc, ngf, inner_ap_nc, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input, inner_ap):
        """Standard forward"""
        return self.model(input, inner_ap)


class MultiCNNMaskBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, branch_num, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(MultiCNNMaskBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        mask_models = []
        for i in range(0, branch_num):
            equalconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3 + 2 * i,
                                  stride=1, padding=1 + i, bias=use_bias)
            equalrelu = nn.LeakyReLU(0.2, True)
            equalnorm = norm_layer(inner_nc)

            mask_model = [equalconv, equalnorm, equalrelu]
            mask_models.append(nn.Sequential(*mask_model))
        self.mask_models = nn.ModuleList(mask_models)

        model = [submodule]
        self.model = nn.Sequential(*model)

    def forward(self, x, mask):
        mask_y = None
        for mask_model in self.mask_models:
            mask_y_branch = mask_model(x) * mask
            if mask_y is None:
                mask_y = mask_y_branch
            else:
                mask_y = torch.cat([mask_y, mask_y_branch], 1)

        return self.model(mask_y)


class MultiCNNMaskRandomBGBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, branch_num, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(MultiCNNMaskRandomBGBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        mask_models = []
        for i in range(0, branch_num):
            equalconv = nn.Conv2d(input_nc, inner_nc, kernel_size=7 + 2 * i,
                                  stride=1, padding=3 + i, bias=use_bias)
            # equalconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3 + 2 * i,
            #                       stride=1, padding=1 + i, bias=use_bias)
            equalrelu = nn.LeakyReLU(0.2, True)
            equalnorm = norm_layer(inner_nc)

            mask_model = [equalconv, equalnorm, equalrelu]
            mask_models.append(nn.Sequential(*mask_model))
        self.mask_models = nn.ModuleList(mask_models)

        shrinkconv = nn.Conv2d(inner_nc * branch_num, branch_num, kernel_size=1,
                              stride=1, padding=0, bias=use_bias)
        shrinkrelu = nn.LeakyReLU(0.2, True)
        shrinknorm = norm_layer(branch_num)

        disperseconv = nn.Conv2d(branch_num, branch_num, kernel_size=11,
                               stride=1, padding=5, bias=use_bias)
        disperserelu = nn.LeakyReLU(0.2, True)
        dispersenorm = norm_layer(branch_num)

        shrinkpart = [shrinkconv, shrinknorm, shrinkrelu]
        dispersepart = [disperseconv, dispersenorm, disperserelu]

        self.shrinkpart = nn.Sequential(*shrinkpart)
        self.dispersepart = nn.Sequential(*dispersepart)

        model = [submodule]
        self.model = nn.Sequential(*model)

    def forward(self, x, mask, random_bg):
        mask_y = None
        for mask_model in self.mask_models:
            mask_y_branch = mask_model(x) * mask
            if mask_y is None:
                mask_y = mask_y_branch
            else:
                mask_y = torch.cat([mask_y, mask_y_branch], 1)

        mask_y = mask_y + random_bg * (1 - mask)
        processed_y = self.shrinkpart(mask_y)

        processed_y = self.random_move_controlling_stick(processed_y, mask)

        processed_y = self.dispersepart(processed_y)
        return self.model(processed_y)

    def random_move_controlling_stick(self, processed_y, mask):
        controlling_stick_gap = 5
        cut_width = 50
        upper_bound = processed_y.shape[3] - 1 - cut_width
        src_pos_x = random.randint(0, upper_bound)
        src_pos_x = src_pos_x - src_pos_x % controlling_stick_gap
        src_pos_y = random.randint(0, upper_bound)

        tag_pos_x = random.randint(0, upper_bound - int(controlling_stick_gap / 2))
        tag_pos_x = tag_pos_x - tag_pos_x % controlling_stick_gap + int(controlling_stick_gap / 2)
        tag_pos_y = random.randint(0, upper_bound)

        ret_y = processed_y.clone()
        ret_y[:, :, tag_pos_y: tag_pos_y + cut_width, tag_pos_x: tag_pos_x + cut_width] += \
            processed_y[:, :, src_pos_y: src_pos_y + cut_width, src_pos_x: src_pos_x + cut_width]
        return ret_y


class MaskCollectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, branch_num, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(MaskCollectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        mask_models = []
        for i in range(0, branch_num):
            equalconv = nn.Conv2d(input_nc, inner_nc, kernel_size=7 + 2 * i,
                                  stride=1, padding=3 + i, bias=use_bias)
            # equalconv = nn.Conv2d(input_nc, inner_nc, kernel_size=3 + 2 * i,
            #                       stride=1, padding=1 + i, bias=use_bias)
            equalrelu = nn.LeakyReLU(0.2, True)
            equalnorm = norm_layer(inner_nc)

            mask_model = [equalconv, equalnorm, equalrelu]
            mask_models.append(nn.Sequential(*mask_model))
        self.mask_models = nn.ModuleList(mask_models)

        shrinkconv = nn.Conv2d(inner_nc * branch_num, outer_nc, kernel_size=1,
                              stride=1, padding=0, bias=use_bias)
        shrinkrelu = nn.LeakyReLU(0.2, True)
        shrinknorm = norm_layer(outer_nc)

        shrinkpart = [shrinkconv, shrinknorm, shrinkrelu]

        self.shrinkpart = nn.Sequential(*shrinkpart)

    def forward(self, x, mask, random_bg):
        mask_y = None
        for mask_model in self.mask_models:
            mask_y_branch = mask_model(x) * mask
            if mask_y is None:
                mask_y = mask_y_branch
            else:
                mask_y = torch.cat([mask_y, mask_y_branch], 1)

        mask_y = mask_y + random_bg * (1 - mask)
        processed_y = self.shrinkpart(mask_y)

        return processed_y


class DisperseBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, branch_num, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(DisperseBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc

        disperseconv = nn.Conv2d(input_nc, branch_num, kernel_size=11,
                               stride=1, padding=5, bias=use_bias)
        disperserelu = nn.LeakyReLU(0.2, True)
        dispersenorm = norm_layer(branch_num)

        dispersepart = [disperseconv, dispersenorm, disperserelu]

        self.dispersepart = nn.Sequential(*dispersepart)

        model = [submodule]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        processed_y = self.dispersepart(x)
        return self.model(processed_y)


class UnetSkipConnectionInnerRandomBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, inner_ap_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            inner_ap_nc -- the number of channels in inner append vector
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionInnerRandomBlock, self).__init__()
        self.outermost = outermost
        self.innermost = innermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model_down = down
            model_sub = submodule
            model_up = up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc + inner_ap_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model_down = down
            model_sub = None
            model_up = up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model_down = down
                model_sub = submodule
                model_up = up + [nn.Dropout(0.5)]
            else:
                model_down = down
                model_sub = submodule
                model_up = up

        self.model_down = nn.Sequential(*model_down)
        if model_sub is not None:
            self.model_sub = model_sub
        self.model_up = nn.Sequential(*model_up)

    def forward(self, x, inner_ap):
        if self.outermost:
            down_out = self.model_down(x)
            sub_out = self.model_sub(down_out, inner_ap)
            return self.model_up(sub_out)
        elif self.innermost:
            down_out = self.model_down(x)
            sub_out = torch.cat([down_out, inner_ap], 1)
            return torch.cat([x, self.model_up(sub_out)], 1)
        else:   # add skip connections
            down_out = self.model_down(x)
            sub_out = self.model_sub(down_out, inner_ap)
            return torch.cat([x, self.model_up(sub_out)], 1)


class DownsamplingResnetBranchGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', n_downsampling=2):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(DownsamplingResnetBranchGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        pre_n_blocks = int(n_blocks * 0.5)
        post_n_blocks = n_blocks - pre_n_blocks

        comp_model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            comp_model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(pre_n_blocks):       # add ResNet blocks

            comp_model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        self.comp_model = nn.Sequential(*comp_model)

    def forward(self, input):
        """Standard forward"""
        return self.comp_model(input)


class UpsamplingResnetBranchGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', n_downsampling=2):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(UpsamplingResnetBranchGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        pre_n_blocks = int(n_blocks * 0.5)
        post_n_blocks = n_blocks - pre_n_blocks

        upsam_branch_model = []
        mult = 2 ** n_downsampling
        for i in range(post_n_blocks):       # add ResNet blocks

            upsam_branch_model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            upsam_branch_model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        upsam_branch_model += [nn.ReflectionPad2d(3)]
        upsam_branch_model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        upsam_branch_model += [nn.Tanh()]

        self.upsam_branch_model = nn.Sequential(*upsam_branch_model)

    def forward(self, input):
        """Standard forward"""
        return self.upsam_branch_model(input)


class LabelBranchGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect', n_downsampling=2):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(LabelBranchGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        mult = 2 ** n_downsampling
        n_downsampling_to_one = 7 - n_downsampling
        hight_delta_branch_model = []
        input_chanel = ngf * mult
        for i in range(n_downsampling_to_one):  # add downsampling layers
            hight_delta_branch_model += [nn.Conv2d(input_chanel, output_nc, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(output_nc),
                      nn.ReLU(True)]
            input_chanel = output_nc
        hight_delta_branch_model += [nn.Conv2d(input_chanel, output_nc, kernel_size=3, stride=2, padding=1, bias=use_bias),
                                     nn.Sigmoid(),
                                     nn.Flatten()]

        self.hight_delta_branch_model = nn.Sequential(*hight_delta_branch_model)

    def forward(self, input):
        """Standard forward"""
        return self.hight_delta_branch_model(input)


class LeakReluResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(LeakReluResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.LeakyReLU(0.2, True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.LeakyReLU(0.2, True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.LeakyReLU(0.2, True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.LeakyReLU(0.2, True)]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
