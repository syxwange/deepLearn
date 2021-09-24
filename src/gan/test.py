import torch
import torch.nn as nn


device = 'cuda' if torch.cuda.is_available() else 'cpu'
DIM = 64 # Model dimensionality
BATCH_SIZE = 50 # Batch size
CRITIC_ITERS = 5 # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10 # Gradient penalty lambda hyperparameter
ITERS = 200000 # How many generator iterations to train for
OUTPUT_DIM = 784 # Number of pixels in MNIST (28*28)

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.preprocess = nn.Sequential(nn.Linear(128, 4*4*4*DIM), nn.ReLU(True),)
        self.block1 = nn.Sequential(nn.ConvTranspose2d(4*DIM, 2*DIM, 5), nn.ReLU(True) )
        self.block2 = nn.Sequential(nn.ConvTranspose2d(2*DIM, DIM, 5), nn.ReLU(True), )
        self.deconv_out = nn.ConvTranspose2d(DIM, 1, 8, stride=2)       
        self.sigmoid = nn.Sigmoid()

    def forward(self, input):
        output = self.preprocess(input)
        output = output.view(-1, 4*DIM, 4, 4)
        #print output.size()
        output = self.block1(output)
        #print output.size()
        output = output[:, :, :7, :7]
        #print output.size()
        output = self.block2(output)
        #print output.size()
        output = self.deconv_out(output)
        output = self.sigmoid(output)
        #print output.size()
        return output.view(-1, OUTPUT_DIM)


G = Generator().to(device)
nose = torch.randn((128),device=device)
out = G(nose)
print(out.shape)