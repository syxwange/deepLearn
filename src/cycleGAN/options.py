import argparse
from pathlib import Path
import time

class Options():
    def __init__(self) -> None:
        self.initialized, self.parser, self.opt= False, None, None

    def initialize(self,parser:argparse.ArgumentParser):
        parser.add_argument('--path_a',  type=str, help='图片A路径')
        parser.add_argument('--path_b',  type=str, help='图片B路径')        
        parser.add_argument('--dataroot',  type=str,help='图片路径一般包含子路径(trainA, trainB, valA, valB, etc)')
        parser.add_argument('--train', action='store_true', default=False , help='是训练还是生产')
        parser.add_argument('--batch_size', type=int, default=1, help='input batch size')
        parser.add_argument('--load_size', type=int, default=286, help='scale images to this size')
        parser.add_argument('--crop_size', type=int, default=256, help='then crop to this size')
        parser.add_argument('--direction', type=str, default='AtoB', help='AtoB or BtoA')
        # model parameters
        parser.add_argument('--input_nc', type=int, default=3, help='#输入图片通道: 3 for RGB and 1 for grayscale')
        parser.add_argument('--output_nc', type=int, default=3, help='# 输出图片通道: 3 for RGB and 1 for grayscale')
        parser.add_argument('--ngf', type=int, default=64, help='# of gen filters in the last conv layer')
        parser.add_argument('--ndf', type=int, default=64, help='# of discrim filters in the first conv layer')
        parser.add_argument('--netD', type=str, default='basic', help='指定鉴别器体系结构[basic | n_layers | pixel]. The basic model is a 70x70 PatchGAN. n_layers allows you to specify the layers in the discriminator')
        parser.add_argument('--netG', type=str, default='resnet_9blocks', help='指定生成器体系结构 [resnet_9blocks | resnet_6blocks | unet_256 | unet_128]')
        parser.add_argument('--n_layers_D', type=int, default=3, help='only used if netD==n_layers')
        parser.add_argument('--norm', type=str, default='instance', help='instance normalization or batch normalization [instance | batch | none]')
        parser.add_argument('--init_type', type=str, default='normal', help='network initialization [normal | xavier | kaiming | orthogonal]')
        parser.add_argument('--init_gain', type=float, default=0.02, help='scaling factor for normal, xavier and orthogonal.')
        parser.add_argument('--no_dropout', action='store_true', help='no dropout for the generator')
        #train parameters
        parser.add_argument('--n_epochs', type=int, default=2, help='number of epochs with the initial learning rate')
        parser.add_argument('--n_epochs_decay', type=int, default=2, help='number of epochs to linearly decay learning rate to zero')
        parser.add_argument('--lr_decay_iters', type=int, default=50, help='multiply by a gamma every lr_decay_iters iterations')
        parser.add_argument('--verbose', action='store_true', help='if specified, print more debugging information')
        parser.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')        
        parser.add_argument('--save_latest_freq', type=int, default=5000, help='frequency of saving the latest results')
        parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
        parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
        parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate for adam')
        parser.add_argument('--preprocess', type=str, default='resize_and_crop', help='scaling and cropping of images at load time [resize_and_crop | crop | scale_width | scale_width_and_crop | none]')
        parser.add_argument('--pool_size', type=int, default=50, help='存储生面器生成的图像缓冲区大小')
        parser.add_argument('--epoch_count', type=int, default=1, help='训练次数')
        parser.add_argument('--gan_mode', type=str, default='lsgan', help='the type of GAN objective. [vanilla| lsgan | wgangp]. vanilla GAN loss is the cross-entropy objective used in the original GAN paper.')
        #cycleGan parameters
        parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='use identity mapping. Setting lambda_identity other than 0 has an effect of scaling the weight of the identity mapping loss. For example, if the weight of the identity loss should be 10 times smaller than the weight of the reconstruction loss, please set lambda_identity = 0.1')
        self.initialized = True
        return parser

    def print_options(self):
        """Print and save options
        #? 保存文件的路径'./checkpoints/cycleGan'
        """
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(self.opt).items()):
            comment = ''
            default = self.parser.get_default(k)
            if v != default:
                comment = f'[default:{ str(default)}]' 
            message += f'{str(k):>25}: { str(v):<30}{comment}\n'
        message += '----------------- End -------------------'
        print(message)

        # save to the disk        
        dateTime =time.strftime("%Y-%m-%d(%H.%M)", time.localtime()) 
        file_name = Path(self.opt.save_dir)/ f'{dateTime}_opt.txt' 
        with open(str(file_name), 'wt') as opt_file:
            opt_file.write(message)
            opt_file.write('\n')

    def parse(self):
        if not self.initialized:  # check if it has been initialized
            parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)

        self.parser = parser
        self.opt = parser.parse_args()
        #多GPU在这里设置
        self.opt.gpu_ids = [0]
        save_dir = Path('./checkpoints/cycleGan/')
        save_dir.mkdir(exist_ok=True)
        self.opt.save_dir = str(save_dir)       
        return self.opt
