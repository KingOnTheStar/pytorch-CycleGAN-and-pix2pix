import torch
import itertools
from .base_model import BaseModel
from . import networks
import kornia
import cv2 as cv
from losses.directional_derivative_loss import DirectionalDerivativeLoss


class Pix2PixDemHidiffGenModel(BaseModel):
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
        parser.set_defaults(norm='batch', netG='unet_256', dataset_mode='aligned')
        # Custom parameters
        parser.add_argument('--background_color', type=int, default=244, help='the background color of input data')
        parser.add_argument('--mask_L1_loss', action='store_true', help='if specified, calculate L1 on mask area only')
        parser.add_argument('--cal_road_perpendicular_line', action='store_true', help='if specified, calculate road perpendicular line')
        parser.add_argument('--cal_water_area_mask', action='store_true', help='if specified, calculate water area mask')
        parser.add_argument('--tv_loss', action='store_true', help='if specified, add total variation regularization term')
        parser.add_argument('--dd_loss', action='store_true', help='if specified, add directional derivative regularization term')
        parser.add_argument('--avghi_loss', action='store_true', help='if specified, add loss to restrain the average height of output dem')
        parser.add_argument('--water_area_loss', action='store_true', help='if specified, add water area loss term')
        parser.add_argument('--add_inner_random', action='store_true', help='if specified, add random number to the inner network')
        parser.add_argument('--add_inner_random_netG', type=str, default='inner_random_net_256', help='netG used with --add_inner_random parameter')
        parser.add_argument('--inner_random_nc', type=int, default=64, help='the channel number of inner_random')
        parser.add_argument('--average_height', type=int, default=-1, help='0-255, the average height of generated dem img, if < 0 ,use the average height of real dem')
        parser.add_argument('--water_area_edge_hidiff', type=int, default=0.1, help='the target height difference of water area and edge')
        parser.add_argument('--gen_height_diff', action='store_true', help='if specified, generate height difference value by network')
        parser.add_argument('--hidiff_downsampling_netG_branch', type=str, default='downsampling_resnet_branch_9blocks', help='netG used with --gen_height_diff parameter')
        parser.add_argument('--hidiff_upsampling_netG_branch', type=str, default='upsampling_resnet_branch_9blocks', help='netG used with --gen_height_diff parameter')
        parser.add_argument('--hidiff_label_netG_branch', type=str, default='label_resnet_branch_9blocks', help='netG used with --gen_height_diff parameter')
        parser.add_argument('--hidiff_output_nc', type=int, default=5, help='the length of output channel of label resnet')
        if is_train:
            parser.set_defaults(pool_size=0, gan_mode='vanilla')
            parser.add_argument('--lambda_L1', type=float, default=100.0, help='weight for L1 loss')
            parser.add_argument('--lambda_vtreg', type=float, default=0.0001, help='weight for TV regularization term')
            parser.add_argument('--lambda_ddreg', type=float, default=1.0, help='weight for DD regularization term')
            parser.add_argument('--lambda_avghi', type=float, default=30.0, help='weight for average height loss')
            parser.add_argument('--lambda_wam', type=float, default=100.0, help='weight for water area and edge loss')

        return parser

    def __init__(self, opt):
        """Initialize the pix2pix class.

        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts will call <BaseModel.get_current_losses>
        self.loss_names = ['G_GAN', 'G_L1', 'D_real', 'D_fake', 'reg', 'other']
        # specify the images you want to save/display. The training/test scripts will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        #self.visual_names = ['fake_B']
        # specify the extra data you want to save
        self.extra_data_names = ['extra_data']
        # Whether use L1 mask loss
        self.mask_loss = opt.mask_L1_loss
        # Whether use variation regularization term
        self.tv_loss = opt.tv_loss
        # Whether use directional derivative term
        self.dd_loss = opt.dd_loss
        # Whether use average height loss
        self.avghi_loss = opt.avghi_loss
        # Whether use water area loss
        self.water_area_loss = opt.water_area_loss
        # Whether add random number
        self.add_inner_random = opt.add_inner_random
        # The target height difference of water area and edge
        self.water_area_edge_hidiff = opt.water_area_edge_hidiff
        # Need generate height difference
        self.gen_height_diff = opt.gen_height_diff
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        if self.isTrain:
            self.model_names = ['G', 'D', 'Branch1', 'Branch2']
        else:  # during test time, only load G
            self.model_names = ['G', 'Branch1', 'Branch2']
        # define networks (both generator and discriminator)
        if self.add_inner_random:
            inner_ap_nc = opt.inner_random_nc + (1 if self.avghi_loss else 0)
            self.netG = networks.define_exp_G(opt.input_nc, opt.output_nc, opt.ngf, opt.add_inner_random_netG, inner_ap_nc=inner_ap_nc,
                                              norm=opt.norm, use_dropout=not opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        elif self.gen_height_diff:
            self.netG = networks.define_exp_G(opt.input_nc, opt.output_nc, opt.ngf, opt.hidiff_downsampling_netG_branch, norm=opt.norm,
                                              use_dropout=not opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
            self.netBranch1 = networks.define_exp_G(opt.input_nc, opt.output_nc, opt.ngf, opt.hidiff_upsampling_netG_branch, norm=opt.norm,
                                                    use_dropout=not opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
            self.netBranch2 = networks.define_exp_G(opt.input_nc, opt.hidiff_output_nc, opt.ngf, opt.hidiff_label_netG_branch, norm=opt.norm,
                                                    use_dropout=not opt.no_dropout, init_type=opt.init_type, init_gain=opt.init_gain, gpu_ids=self.gpu_ids)
        else:
            self.netG = networks.define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.netG, opt.norm,
                                          not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:  # define a discriminator; conditional GANs need to take both input and output images; Therefore, #channels for D is input_nc + output_nc
            self.netD = networks.define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.netD,
                                          opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionLabel = torch.nn.CrossEntropyLoss()
            self.tv_regularization_term = kornia.losses.TotalVariation()
            self.dd_regularization_term = DirectionalDerivativeLoss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            if self.gen_height_diff:
                self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG.parameters(), self.netBranch1.parameters(), self.netBranch2.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            else:
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_preinput(self, preinput):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A_norm = preinput['A_norm' if AtoB else 'B_norm']
        self.real_B_norm = preinput['B_norm' if AtoB else 'A_norm']
        self.fake_B_norm = self.real_B_norm

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap images in domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        if self.mask_loss:
            self.real_A_mask = input['A_mask' if AtoB else 'B_mask'].to(self.device)
        if self.dd_loss:
            self.rpl_img = input['rpl'].to(self.device)
        if self.water_area_loss:
            self.wam_area_img = input['wam_area'].to(self.device)
            self.wam_edge_img = input['wam_edge'].to(self.device)
        if self.add_inner_random:
            self.inner_random = input['inner_random'].to(self.device)
        if self.avghi_loss:
            self.avg_height = input['avg_height'].to(self.device)
        if self.gen_height_diff:
            self.hidiff_idx = input['hidiff_idx'].to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        if self.gen_height_diff:
            netG_out = self.netG(self.real_A)
            self.fake_B = self.netBranch1(netG_out)
            self.hidiff = self.netBranch2(netG_out)
            self.extra_data = {'hidiff': self.hidiff}
        else:
            if not self.avghi_loss and self.add_inner_random:
                self.fake_B = self.netG(self.real_A, self.inner_random)  # G(A)
            elif self.avghi_loss and self.add_inner_random:
                inner_ap = torch.cat((self.inner_random, self.avg_height), 1)
                self.fake_B = self.netG(self.real_A, inner_ap)  # G(A)
            else:
                self.fake_B = self.netG(self.real_A)  # G(A)

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
            self.loss_G_L1 = self.criterionL1(self.real_A_mask * self.fake_B,
                                              self.real_A_mask * self.real_B) * self.opt.lambda_L1
        else:
            self.loss_G_L1 = self.criterionL1(self.fake_B, self.real_B) * self.opt.lambda_L1

        # Regularization term
        self.loss_reg = 0
        if self.tv_loss:
            loss_tv = self.tv_regularization_term(self.fake_B) * self.opt.lambda_vtreg
            self.loss_reg += loss_tv
        if self.dd_loss:
            loss_dd = self.dd_regularization_term(self.fake_B, self.rpl_img) * self.opt.lambda_ddreg
            self.loss_reg += loss_dd
        if self.avghi_loss:
            loss_avghi = (torch.mean(self.avg_height.abs()) - torch.mean(self.fake_B.abs())).abs() * self.opt.lambda_avghi
            self.loss_reg += loss_avghi
        if self.water_area_loss:
            wam_area_pixel_num = torch.sum(self.wam_area_img)
            wam_edge_pixel_num = torch.sum(self.wam_edge_img)
            wam_area_hi = torch.sum(self.fake_B * self.wam_area_img) / (wam_area_pixel_num + 1e-27)
            wam_edge_hi = torch.sum(self.fake_B * self.wam_edge_img) / (wam_edge_pixel_num + 1e-27)
            loss_wam = ((wam_edge_hi - self.water_area_edge_hidiff) - wam_area_hi).abs() * self.opt.lambda_wam
            loss_wam *= wam_area_pixel_num / (wam_area_pixel_num + 1e-27) * wam_edge_pixel_num / (wam_edge_pixel_num + 1e-27)
            self.loss_reg += loss_wam

        self.loss_other = 0
        if self.gen_height_diff:
            loss_hidiff = self.criterionLabel(self.hidiff, self.hidiff_idx.long())
            self.loss_other += loss_hidiff
        # combine loss and calculate gradients
        self.loss_G = self.loss_G_GAN + self.loss_G_L1 + self.loss_reg + self.loss_other
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
