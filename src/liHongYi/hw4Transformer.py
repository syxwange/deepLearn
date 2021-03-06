import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset ,DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
from torch.optim import AdamW
                  
"""#! 任务描述
#? 对特定特征的说话人进行分类。
#! 主要目标：学习如何使用transformer。

#! 对于transformer，学习率计划的设计不同于CNN
- 以前的工作表明，warmup of learning rate对于transformer的训练模型是有用的。
- The warmup schedule
  - Set learning rate to 0 in the beginning.
  - The learning rate increases linearly from 0 to initial learning rate during warmup period.    [extended_summary]
"""


class myDataset(Dataset):
    def __init__(self, data_dir, segment_len=128):
        self.data_dir  ,self.segment_len=data_dir ,segment_len 
        
        speaker2id = json.load(open(data_dir+"/mapping.json"))["speaker2id"]    
        metadata = json.load(open(data_dir+"/metadata.json"))["speakers"] 
        self.speaker_num = len(metadata.keys())
        self.data = []
        for speaker in metadata.keys():
            for utterances in metadata[speaker]:
                self.data.append([utterances["feature_path"], speaker2id[speaker]])
            
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, index):
        feat_path, speaker = self.data[index]
        # Load preprocessed mel-spectrogram.
        mel = torch.load(os.path.join(self.data_dir, feat_path))
        #最大长度限制在segment_len ;max mel [128*40]
        if len(mel) > self.segment_len:      
            start = random.randint(0, len(mel) - self.segment_len)      
            mel = torch.FloatTensor(mel[start:start+self.segment_len])
        else:
            mel = torch.FloatTensor(mel)    
        speaker = torch.FloatTensor([speaker]).long()
        return mel, speaker
     
    def get_speaker_number(self):
        return self.speaker_num

def collate_batch(batch):
    # Process features within a batch.
    """Collate a batch of data."""
    mel, speaker = zip(*batch)
    # Because we train the model batch by batch, we need to pad the features in the same batch to make their lengths the same.
    mel = pad_sequence(mel, batch_first=True, padding_value=-20)    # pad log 10^(-20) which is very small value.
    # mel: (batch size, length, 40)
    return mel, torch.FloatTensor(speaker).long()

def get_dataloader(data_dir, batch_size, n_workers):
    """生成数据加载器"""
    dataset = myDataset(data_dir)
    speaker_num = dataset.get_speaker_number()
    # 将数据集拆分为训练数据集和验证数据集
    trainlen = int(0.9 * len(dataset))  
    trainset, validset = random_split(dataset, [trainlen, len(dataset) - trainlen])

    train_loader = DataLoader(trainset, batch_size=batch_size,  shuffle=True,  drop_last=True,
        num_workers=n_workers,  pin_memory=True,   collate_fn=collate_batch,  )
    valid_loader = DataLoader( validset,  batch_size=batch_size,  num_workers=n_workers,
        drop_last=True,  pin_memory=True,  collate_fn=collate_batch, )
    return train_loader, valid_loader, speaker_num

class Classifier(nn.Module):
    def __init__(self, d_model=80, n_spks=600, dropout=0.1):
        super().__init__()
    # Project the dimension of features from that of input into d_model.
        self.prenet = nn.Linear(40, d_model)
        # TODO:
        #   Change Transformer to Conformer.
        #   https://arxiv.org/abs/2005.08100
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, dim_feedforward=256, nhead=2)
    # self.encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=2)

    # Project the the dimension of features from d_model into speaker nums.
        self.pred_layer = nn.Sequential( nn.Linear(d_model, d_model), nn.ReLU(),  nn.Linear(d_model, n_spks), )
            
    def forward(self, mels):
        """
        args:      mels: (batch size, length, 40)
        return:     out: (batch size, n_spks)
        """
        # out: (batch size, length, d_model)
        out = self.prenet(mels)
        # out: (length, batch size, d_model)
        out = out.permute(1, 0, 2)
        # The encoder layer expect features in the shape of (length, batch size, d_model).
        out = self.encoder_layer(out)
        # out: (batch size, length, d_model)
        out = out.transpose(0, 1)
        # mean pooling
        stats = out.mean(dim=1)
        # out: (batch, n_spks)
        out = self.pred_layer(stats)
        return out

#! 对于transformer，学习率计划的设计不同于CNN
def get_cosine_schedule_with_warmup(  optimizer: Optimizer,
  num_warmup_steps: int,  num_training_steps: int,  num_cycles: float = 0.5,  last_epoch: int = -1,):
    """
    #! Create a schedule with a learning rate that decreases following the values of the cosine function between the
    initial lr set in the optimizer to 0, after a warmup period during which it increases linearly between 0 and the
    initial lr set in the optimizer.
    Args:
        optimizer (:class:`~torch.optim.Optimizer`):
        The optimizer for which to schedule the learning rate.
        num_warmup_steps (:obj:`int`):
        The number of steps for the warmup phase.
        num_training_steps (:obj:`int`):
        The total number of training steps.
        num_cycles (:obj:`float`, `optional`, defaults to 0.5):
        The number of waves in the cosine schedule (the defaults is to just decrease from the max value to 0
        following a half-cosine).
        last_epoch (:obj:`int`, `optional`, defaults to -1):
        The index of the last epoch when resuming training.
    Return:
        :obj:`torch.optim.lr_scheduler.LambdaLR` with the appropriate schedule.
    """
    def lr_lambda(current_step):
    # Warmup
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
    # decadence
        progress = float(current_step - num_warmup_steps) / float( max(1, num_training_steps - num_warmup_steps) )
        return max( 0.0, 0.5 * (1.0 + math.cos(math.pi * float(num_cycles) * 2.0 * progress))  )

    return LambdaLR(optimizer, lr_lambda, last_epoch)

def model_fn(batch, model, criterion, device):
    """Forward a batch through the model."""
    mels, labels = batch
    mels ,labels = mels.to(device), labels.to(device)
    outs = model(mels)
    loss = criterion(outs, labels)
    # Get the speaker id with highest probability.
    preds = outs.argmax(1)
    # Compute accuracy.
    accuracy = torch.mean((preds == labels).float())
    return loss, accuracy

def valid(dataloader, model, criterion, device): 
    """Validate on validation set."""
    model.eval()
    running_loss = 0.0
    running_accuracy = 0.0
    pbar = tqdm(total=len(dataloader.dataset), ncols=0, desc="Valid", unit=" uttr")

    for i, batch in enumerate(dataloader):
        with torch.no_grad():
            loss, accuracy = model_fn(batch, model, criterion, device)
            running_loss += loss.item()
            running_accuracy += accuracy.item()

        pbar.update(dataloader.batch_size)
        pbar.set_postfix(loss=f"{running_loss / (i+1):.2f}",  accuracy=f"{running_accuracy / (i+1):.2f}", )

    pbar.close()
    model.train()
    return running_accuracy / len(dataloader)

def parse_args():
    """arguments"""
    config = {"data_dir": "./data/Dataset",  "save_path": "model.ckpt",  "batch_size": 32,   "n_workers": 8,
        "valid_steps": 2000,   "warmup_steps": 1000,   "save_steps": 10000,   "total_steps": 70000, }
    return config

def Train( data_dir, save_path, batch_size, n_workers, valid_steps, warmup_steps, total_steps, save_steps,):
    """Main function."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[Info]: Use {device} now!")

    train_loader, valid_loader, speaker_num = get_dataloader(data_dir, batch_size, n_workers)
    train_iterator = iter(train_loader)
    print(f"[Info]: Finish loading data!",flush = True)

    model = Classifier(n_spks=speaker_num).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=1e-3)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    print(f"[Info]: Finish creating model!",flush = True)

    best_accuracy = -1.0
    best_state_dict = None

    pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

    for step in range(total_steps):
        # Get data
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)

        loss, accuracy = model_fn(batch, model, criterion, device)
        batch_loss = loss.item()
        batch_accuracy = accuracy.item()

        # Updata model
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
    
        # Log
        pbar.update()
        pbar.set_postfix(loss=f"{batch_loss:.2f}", accuracy=f"{batch_accuracy:.2f}",   step=step + 1,   )

        # Do validation
        if (step + 1) % valid_steps == 0:
            pbar.close()
            valid_accuracy = valid(valid_loader, model, criterion, device)

            # keep the best model
            if valid_accuracy > best_accuracy:
                best_accuracy = valid_accuracy
                best_state_dict = model.state_dict()

            pbar = tqdm(total=valid_steps, ncols=0, desc="Train", unit=" step")

        # Save the best model so far.
        if (step + 1) % save_steps == 0 and best_state_dict is not None:
            torch.save(best_state_dict, save_path)
            pbar.write(f"Step {step + 1}, best model saved. (accuracy={best_accuracy:.4f})")

    pbar.close()

if __name__=="__main__":
    Train(**parse_args())


