import torch
import itertools
from .base_model import BaseModel
from . import networks
from data import create_dataset, create_dataset_params
import cv2 as cv


class Pix2PixCtrlstickBgfusionModel(BaseModel):
    """ This class implements the pix2pix model, for learning a mapping from input images to output images given paired data.

    The model training requires '--dataset_mode aligned' dataset.
    By default, it uses a '--netG unet256' U-Net generator,
    a '--netD basic' discriminator (PatchGAN),
    and a '--gan_mode' vanilla GAN loss (the cross-entropy objective used in the orignal GAN paper).

    pix2pix paper: https://arxiv.org/pdf/1611.07004.pdf
    """
    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        For pix2pix, we do not use image buffer
        The training objective is: GAN Loss + lambda_L1 * ||G(A)-B||_1
        By default, we use vanilla GAN loss, UNet with batchnorm, and aligned datasets.
        """
        # changing the default values to match the pix2pix paper (https://phillipi.github.io/pix2pix/)
        parser.set_defaults(norm='batch', netG='post_mask_unet_256', dataset_mode='aligned')
        # Custom parameters
        parser.add_argument('--netStage1', type=str, default='mask_collection_256', help='specify generator architecture in stage 1')
        parser.add_argument('--mask_L1_loss', action='store_true', help='if specified, calculate L1 on mask area only')
        parser.add_argument('--random_background', action='store_true', help='if specified, fill random number in area without mask')
        parser.add_argument('--net_branch_num', type=int, default=3, help='the branch num of network')
        parser.add_argument('--enrich_background', action='store_true', help='if specified, generate more data in background')
        parser.add_argument('--background_color', type=int, default=244, help='the background color of input data')
        parser.add_argument('--use_controlling_stick', action='store_true', help='if specified, use controlling stick to generate the image')
        parser.add_argument('--fusion_controlling_stick', action='store_true', help='if specified, create fusion controlling stick')
        parser.add_argument('--extract_dataroot', type=str, default='./datasets/skestrt', help='the data set used to extract additional image')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # Whether use L1 mask loss
        self.mask_loss = opt.mask_L1_loss
        # Whether set background random
        self.random_background = opt.random_background
        # The color of background
        self.normal_bg_color = opt.background_color / 255.0
        # Need enrich background
        self.enrich_background = opt.enrich_background
        # Use controlling stick
        self.use_controlling_stick = opt.use_controlling_stick
        # Use fusion controlling stick
        self.fusion_controlling_stick = opt.fusion_controlling_stick
        # Create data set for fusion
        ext_params = {'batch_size': 1, 'transmit_params': True, 'mini_out': True,
                      'use_extract_dataroot': True}
        self.dataset = create_dataset_params(opt, ext_params)
        self.dataset_iter = iter(self.dataset)
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['Stage1', 'G', 'D']
        else:  # during test time, only load G
            self.model_names = ['Stage1', 'G']

        # define networks stage 1 generator
        self.netStage1 = networks.define_exp_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netStage1, opt.net_branch_num, opt.norm,
                                               not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        # define networks generator
        self.netG = networks.define_exp_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.net_branch_num, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netStage1.parameters(), self.netG.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.real_A_mask = input['A_mask' if AtoB else 'B_mask'].to(self.device)
        if self.mask_loss:
            self.real_B_mask = input['B_mask' if AtoB else 'A_mask'].to(self.device)
            self.real_AB_mask = self.real_A_mask
            self.real_AB_mask[self.real_B_mask >= 1] = 1
        if self.random_background:
            self.random_bg = input['random_bg'].to(self.device)
        if self.use_controlling_stick:
            self.controlling_stick = input['controlling_stick'].to(self.device)
            if self.fusion_controlling_stick:
                self.fusion_ctrlstick = input['fusion_ctrlstick'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.get_fusion_data()

    def get_fusion_data(self):
        AtoB = self.opt.direction == 'AtoB'
        data = next(self.dataset_iter, None)
        if data is None:
            self.dataset_iter = iter(self.dataset)
            data = next(self.dataset_iter)
        self.fusion_A = data['A' if AtoB else 'B'].to(self.device)
        self.fusion_A_mask = data['A_mask' if AtoB else 'B_mask'].to(self.device)
        if self.mask_loss:
            self.fusion_B_mask = data['B_mask' if AtoB else 'A_mask'].to(self.device)
            self.fusion_AB_mask = self.fusion_A_mask
            self.fusion_AB_mask[self.fusion_B_mask >= 1] = 1
            self.real_AB_mask = self.real_AB_mask + self.fusion_AB_mask
            self.real_AB_mask[self.real_AB_mask >= 2] = 0

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.random_background and not self.use_controlling_stick:
            self.fake_B = self.netG(self.real_A, self.real_A_mask, self.random_bg)  # G(A)
        elif self.random_background and self.use_controlling_stick:
            self.ctrl_B = self.netStage1(self.real_A, self.controlling_stick, self.random_bg)
            fusion_B = self.netStage1(self.fusion_A, self.fusion_ctrlstick, self.random_bg)
            self.ctrl_B = self.ctrl_B + fusion_B
            self.fake_B = self.netG(self.ctrl_B)  # G(A)
        else:
            self.fake_B = self.netG(self.real_A, self.real_A_mask)  # G(A)

    def backward_D(self):
        """Calculate GAN loss for the discriminator"""
        # Fake; stop backprop to the generator by detaching fake_B
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)  # we use conditional GANs; we need to feed both input and output to the discriminator
        pred_fake = self.netD(fake_AB.detach())
        self.loss_D_fake = self.criterionGAN(pred_fake, False)
        # Real
        real_AB = torch.cat((self.real_A, self.real_B), 1)
        pred_real = self.netD(real_AB)
        self.loss_D_real = self.criterionGAN(pred_real, True)
        # combine loss and calculate gradients
        self.loss_D = (self.loss_D_fake + self.loss_D_real) * 0.5
        self.loss_D.backward()

    def backward_G(self):
        """Calculate GAN and L1 loss for the generator"""
        # First, G(A) should fake the discriminator
        fake_AB = torch.cat((self.real_A, self.fake_B), 1)
        pred_fake = self.netD(fake_AB)
        self.loss_G_GAN = self.criterionGAN(pred_fake, True)
        # Second, G(A) = B
        if self.mask_loss:
            self.loss_G_L1 = self.criterionL1(self.real_AB_mask * self.fake_B,
                                              self.real_AB_mask * self.real_B) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1
        # Third, BGMask(G(A)) should not only like background
        if self.enrich_background:
            # fake_bg = (1 - self.real_AB_mask) * self.fake_B + self.real_AB_mask * self.normal_bg_color
            # fake_Abg = torch.cat((self.real_A, fake_bg), 1)
            # pred_fake_bg = self.netD(fake_Abg)
            # self.loss_G_bg_GAN = self.criterionGAN(pred_fake_bg, True)
            # diff_gap = 0
            # diff = torch.sum(torch.pow(fake_bg - self.normal_bg_color, 2))
            # self.loss_G_bg_GAN += (diff - diff_gap) * 0.001
            pass
        else:
            self.loss_G_bg_GAN = 0
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1
        self.loss_G.backward()

    def optimize_parameters(self):
        self.forward()                   # compute fake images: G(A)
        # update D
        self.set_requires_grad(self.netD, True)  # enable backprop for D
        self.optimizer_D.zero_grad()     # set D's gradients to zero
        self.backward_D()                # calculate gradients for D
        self.optimizer_D.step()          # update D's weights
        # update G
        self.set_requires_grad(self.netD, False)  # D requires no gradients when optimizing G
        self.optimizer_G.zero_grad()        # set G's gradients to zero
        self.backward_G()                   # calculate graidents for G
        self.optimizer_G.step()             # udpate G's weights
