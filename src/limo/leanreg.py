
from IPython import get_ipython

from pathlib import Path
import torch
from d2l import torch as d2l
get_ipython().run_line_magic('matplotlib', 'inline')
import random


def synthetic_data(w,b,num):
    x = torch.normal(0,1,(num,len(w)))
    y = torch.matmul(x,w) + b
    y += torch.normal(0,0.01,y.shape)
    return x,y.reshape((-1,1))

ture_w = torch.tensor([2,-3.4])
true_b = 4.2
features, labels = synthetic_data(ture_w,true_b,1000)


d2l.set_figsize()
d2l.plt.scatter(features[:,1].numpy(),labels.numpy(),1)


def data_iter(batch_size,features,labels):
    num = len(features)
    indices = list(range(num))
    random.shuffle(indices)
    for i in range(0,num,batch_size):
        batch_indices = indices[i:min(i+batch_size,num)]
        yield features[batch_indices],labels[batch_indices]


w = torch.normal(0,0.01,size=(2,1),requires_grad=True)
b = torch.zeros(1,requires_grad=True)

def linreg(x,w,b):
    return torch.matmul(x,w)+b

def squared_loss(y_hat,y):
    return (y-y_hat.reshape(y.shape))**2/2

def sgd(params,lr,batch_size):
    with torch.no_grad():
        for param in params:
            param -= lr*param.grad / batch_size
            param.grad.zero_()


lr ,batch_size= 0.03 ,10
epochs = 3
net = linreg
loss = squared_loss

for epoch in range(epochs):
    for x,y in data_iter(batch_size,features,labels):
        l = loss(net(x,w,b),y)
        l.sum().backward()
        sgd([w,b],lr,batch_size)
    with torch.no_grad():
        train = loss(net(features,w,b),labels)
        print(f'epoch:{epoch},loss:{train.mean():.9f}')

