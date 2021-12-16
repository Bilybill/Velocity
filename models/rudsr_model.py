import torch
from .base_model import BaseModel
from .m_rudsr import MRUDSR
from . import networks
import math
import torchsnooper
import torch.nn.functional as F

class RudsrModel(BaseModel):
    """ This TesteModel can be used to generate CycleGAN results for only one direction.
    This model will automatically set '--dataset_mode single', which only loads the images from one collection.

    See the test instruction for more details.
    """
    @staticmethod
    def modify_commandline_options(parser, is_train = True):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.

        The model can only be used during test time. It requires '--dataset_mode single'.
        You need to specify the network using the option '--model_suffix'.
        """
        
        parser.add_argument('--mid_channels', type=int, default=16, help='channles in the middle network')
        parser.add_argument('--num_blocks', type=int, default=30, help='number of network blocks')
        parser.add_argument('--optim', type=str, default='adam', help='choose optimizer')

        return parser

    def __init__(self, opt):
        """Initialize the rudsr class.
        Parameters:
            opt (Option class)-- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseModel.__init__(self, opt)
        # specify the training losses you want to print out. The training/test scripts  will call <BaseModel.get_current_losses>
        self.loss_names = ['L1_loss']
        # specify the images you want to save/display. The training/test scripts  will call <BaseModel.get_current_visuals>
        self.visual_names = ['real_A', 'fake_B', 'real_B']
        # specify the models you want to save to the disk. The training/test scripts will call <BaseModel.save_networks> and <BaseModel.load_networks>
        self.model_names = ['G']
        
        if opt.netG == 'mrudsr':
            self.netG = networks.mrudsr_model(opt.input_nc, opt.output_nc, opt.mid_channels, opt.num_blocks, opt.init_type, opt.init_gain, self.gpu_ids)
        elif opt.netG == 'unet_128':
            self.netG = networks.define_G(opt.input_nc,opt.output_nc, opt.ngf, 'unet_128', 'instance', not opt.no_dropout, opt.init_type, opt.init_gain, self.gpu_ids)

        if self.isTrain:
            self.criterionL1 = torch.nn.SmoothL1Loss()
            # self.criterionL1 = torch.nn.MSELoss()
            if opt.optim == 'adam':
                self.optimizer_G = torch.optim.Adam(self.netG.parameters(), lr = opt.lr, betas = (opt.beta1, 0.999))
            elif opt.optim == 'sgd':
                self.optimizer_G = torch.optim.SGD(self.netG.parameters(), lr = opt.lr, weight_decay = 1e-5, momentum = 0.9)
            else:
                raise ValueError('unkown optim type')
            self.optimizers.append(self.optimizer_G)

    def set_input(self, input):
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
    
    def pad(self, x):
        def floor_ceil(n):
            return math.floor(n), math.ceil(n)
        b, c, h, w = x.shape
        w_mult = ((w - 1) | 31) + 1
        h_mult = ((h - 1) | 31) + 1
        w_pad = floor_ceil((w_mult - w) / 2)
        h_pad = floor_ceil((h_mult - h) / 2)
        x = F.pad(x, w_pad + h_pad, mode = 'replicate')
        return x, (h_pad, w_pad, h_mult, w_mult)

    def unpad(self, x, h_pad, w_pad, h_mult, w_mult):
        return x[..., h_pad[0]:h_mult - h_pad[1], w_pad[0]:w_mult - w_pad[1]]
    
    def forward(self):
        """Run forward pass."""
        y, pad_size = self.pad(self.real_A)
        self.real_A = y
        self.fake_B = self.netG(self.real_A)  # G(A)
        self.fake_B = self.unpad(self.fake_B, *pad_size)
        self.real_A = self.unpad(self.real_A, *pad_size)
        # print('-'*10,'this is fake B max value:',torch.max(self.fake_B),'this is fake B min values:',torch.min(self.fake_B))
    
    def optimize_parameters(self, writer, total_iters):
        self.forward()
        self.loss_L1_loss = self.criterionL1(self.fake_B,self.real_B)
        self.optimizer_G.zero_grad()
        self.loss_L1_loss.backward()
        self.optimizer_G.step()