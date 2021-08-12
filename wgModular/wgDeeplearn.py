
from torch.utils.data import ConcatDataset, DataLoader, Subset,Dataset
import torch
from tqdm.auto import tqdm

class DeeplearnTrain:
    def __init__(self,model,trainSet,validSet,opt=None,lossFunc=None,batchSize=56,epochs=100) -> None:
        self.model,self.opt,self.lossFunc,self.batchSize,self.epochs = model,opt,lossFunc,batchSize,epochs
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.trainSet ,self.validSet= trainSet,validSet
        self.pm = True if self.device=="cuda:0" else False   #如果是GPU 把图片加入到CUDA中的固定内存
        self.trainLoader = DataLoader(self.trainSet,batch_size=self.batchSize,shuffle=True,num_workers=8, pin_memory=self.pm)
        self.validLoader = DataLoader(self.validSet,batch_size=self.batchSize,shuffle=True,num_workers=8, pin_memory=self.pm)
        self.trainLossList ,self.trainAccList,self.validLossList,self.validAccList = [],[],[],[]
        self.bestAcc = 0.5
       
    class CustomSubset(Dataset):
        #自定义DataSet,用于半监督学习未标签的图片，在训练后生成伪标签后组成数据集
        def __init__(self, dataset,  labels):
            self.dataset = dataset
            self.targets = labels
        def __getitem__(self, idx):
            image = self.dataset[idx][0]
            target = self.targets[idx]
            return (image, target)

        def __len__(self):
            return len(self.targets)
            
    def getPseudoLabels(self, unlabelSet,threshold=0.65):
    # 此函数使用给定的模型生成数据集的伪标签
    # 它返回一个DatasetFolder实例，其中包含预测可信度超过给定阈值的图像    
        data_loader = DataLoader(unlabelSet, batch_size=self.batchSize, shuffle=False) 
        self.model.eval()    
        softmax = torch.nn.Softmax(dim=-1)      
        chioce ,labels ,count= [], [], 0
        for batch in data_loader:
            img, _ = batch      
            with torch.no_grad():
                logits = self.model(img.to(self.device))
            # Obtain the probability distributions by applying softmax on logits.
            probs = softmax(logits)
            # ---------- TODO ----------
            for prob in probs:
                print(torch.max(prob))
                if   torch.max(prob) > threshold:
                    chioce.append(count)
                    labels.append(torch.argmax(prob))                
                count +=1

        subDataSet = Subset(unlabelSet,chioce)
        dataset = DeeplearnTrain.CustomSubset(dataset=subDataSet,indices=chioce,labels=labels)    
        return dataset

    def train(self,doSemi=False, unlabelSet=None): 
        for epoch in  tqdm(range(self.epochs)):    
            if doSemi:
                pseudoSet = self.getPseudoLabels(unlabelSet)
                concatSet = ConcatDataset([self.trainSet,pseudoSet])
                self.trainLoader = DataLoader(concatSet,batch_size=self.batchSize,shuffle=True,num_workers=8,pin_memory=self.pm)

            self.model.train()
            trainLoss, trainAcc= [],[]
            for imgs,labels in self.trainLoader:
                ret = self.model(imgs.to(self.device))
                loss = self.lossFunc(ret,labels.to(self.device))
                self.opt.zero_grad()
                loss.backward()
                #?梯度减切Gradient Clip设置一个梯度减切的阈值，如果在更新梯度的时候，
                #? 梯度超过这个阈值，则会将其限制在这个范围之内，防止梯度爆炸。
                gradNorm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=10)
                self.opt.step()
                acc = (torch.argmax(ret,dim=1)==labels.to(self.device)).float().mean()
                trainLoss.append(loss)
                trainAcc.append(acc)

            trainLoss,trainAcc = sum(trainLoss)/len(trainLoss),sum(trainAcc)/len(trainAcc)  
            self.trainLossList.append(trainLoss)
            self.trainAccList.append(trainAcc)

            self.model.eval()
            validLoss ,validAcc= [],[]
            for imgs,labels in self.validLoader:
                with torch.no_grad():
                    ret = self.model(imgs.to(self.device))
                    loss = self.lossFunc(ret,labels.to(self.device))
                acc = (torch.argmax(ret,dim=1)==labels.to(self.device)).float().mean()
                validLoss.append(loss)
                validAcc.append(acc)

            validLoss ,validAcc= sum(validLoss)/len(validLoss),sum(validAcc)/len(validAcc)     
            self.validLossList.append(validLoss)
            self.validAccList.append(validAcc)

            if epoch%5==0:
                print(f"Train--epoch:{epoch}/{self.epochs}--loss={trainLoss:.5f}--acc={trainAcc:.5f}")
                print(f"*****************Valid--loss={validLoss:.5f}--acc={validAcc:.5f}")

            if validAcc > self.bestAcc:
                self.bestAcc = validAcc
                torch.save(self.model.state_dict(),"./bestAcc.ckpt")     

