import torch
from PIL import Image
import torchvision.transforms as transforms
import options
import time
import cycleGanModel
import torchvision

def getSingleData(opt):
    A_img = Image.open(opt.path_a).convert('RGB')
    B_img = Image.open(opt.path_b).convert('RGB')
    osize = [opt.load_size, opt.load_size]
    tfm = transforms.Compose([
        #transforms.Resize(osize, Image.BICUBIC),
        transforms.RandomCrop(opt.crop_size),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
    A =tfm(A_img)
    B = tfm(B_img)
    A = torch.unsqueeze(A,0)
    B = torch.unsqueeze(B,0)  
    return {'A':A,'B':B}


if __name__=='__main__':
    para = options.Options()
    opt = para.parse()
    opt.train = True
    opt.lr_policy='step'
    #opt.netG = 'resnet_6blocks'
    #opt.train =False
    opt.lr_decay_iters=300
    opt.no_dropout=True
    if opt.path_a is None:
        opt.path_a ='./data/a.jpg'
        opt.path_b = './data/b.jpg'
    file  =opt.save_dir+'/horse2zebra.pth'

    
    dataSet = getSingleData(opt)
    para.print_options()    
    model = cycleGanModel.CycleGANModel(opt)  
    
    model.setup(opt)  
    model.load_net()
    model.save()
    total_iters = 0
    for epoch in range(2000):
        data = dataSet
        model.set_input(data)
        model.optimize_parameters()
        model.update_learning_rate() 

        if epoch%20==0:
            f_imgs_sample = (model.fake_B.data + 1) / 2.0
            filename =opt.save_dir+ f'Epoch_{epoch+1:03d}a.jpg'
            torchvision.utils.save_image(f_imgs_sample, filename)
            f_imgs_sample = (model.fake_A.data + 1) / 2.0
            filename =opt.save_dir+ f'Epoch_{epoch+1:03d}b.jpg'
            torchvision.utils.save_image(f_imgs_sample, filename)
            print(f' | Save some samples to {filename}.')   
            
    



