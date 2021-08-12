
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder 
from torch.utils.data import ConcatDataset, DataLoader, Subset,Dataset
import torch
from tqdm.auto import tqdm
from PIL import Image
import torchvision

from wgModular.wgDeeplearn import DeeplearnTrain



trainTfm =transforms.Compose([transforms.Resize((128,128)),
transforms.RandomRotation(45),
transforms.RandomHorizontalFlip(0.5),
transforms.RandomVerticalFlip(0.5),
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
testTfm =transforms.Compose([transforms.Resize((128,128)), 
transforms.ToTensor(),
transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


batch_size = 128
device = "cuda:0" if torch.cuda.is_available() else "cpu"
trainSet = ImageFolder("./data/food-11/training/labeled/",transform=trainTfm)
unlabelSet = ImageFolder("./data/food-11/training/unlabeled/",transform=trainTfm)
validSet = ImageFolder("./data/food-11/validation/",transform=testTfm)
testSet = ImageFolder("./data/food-11/testing/",transform=testTfm)

pm = True if device=="cuda:0" else False   #如果是GPU 把图片加入到CUDA中的固定内存
trainLoader = DataLoader(trainSet,batch_size=batch_size,shuffle=True,num_workers=8, pin_memory=pm)
validLoader = DataLoader(validSet,batch_size=batch_size,shuffle=True,num_workers=8, pin_memory=pm)
testLoader = DataLoader(testSet,batch_size=batch_size,num_workers=8, pin_memory=pm)



class CNNModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.cnnLayers = torch.nn.Sequential(
            torch.nn.Conv2d(3, 64, 3, 1, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2, 0),

            torch.nn.Conv2d(64, 128, 3, 1, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(2, 2, 0),

            torch.nn.Conv2d(128, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.ReLU(),
            torch.nn.MaxPool2d(4, 4, 0),
        )
        self.fcLayers = torch.nn.Sequential(
            torch.nn.Linear(256 * 8 * 8, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 11)
        )

    def forward(self, x):      
        x = self.cnnLayers(x)
        x = x.flatten(1)
        x = self.fcLayers(x)
        return x


class myVgg(torch.nn.Module):
    def __init__(self):
        super().__init__()
        vgg = torchvision.models.vgg11_bn()
        self.layers =vgg.features
        self.fc = vgg.classifier
        self.fc._modules['0'] =torch.nn.Linear(in_features=512*4*4, out_features=512, bias=True)
        self.fc._modules['3'] =torch.nn.Linear(in_features=512, out_features=512, bias=True)
        self.fc._modules['6'] =torch.nn.Linear(in_features=512, out_features=11, bias=True)

    def forward(self,x):
        x =self.layers(x)
        x= x.flatten(1)
        x = self.fc(x)
        return x



# bestModel  =torch.load("./data/bestAcc.ckpt",map_location=device)
# model.load_state_dict(bestModel)
vgg = myVgg().to(device)
opt = torch.optim.Adam(vgg.parameters(),lr=0.001)
lossFunc = torch.nn.CrossEntropyLoss()

train = DeeplearnTrain(model=vgg,trainSet=trainSet,validSet=validSet,opt=opt,lossFunc=lossFunc,batchSize=128,epochs=50)
#train.train(doSemi=True,unlabelSet= unlabelSet)
if __name__=="__main__":
    train.train()



