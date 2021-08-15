

import os
import json
import torch
import random
from pathlib import Path
from torch.utils.data import Dataset ,DataLoader, random_split
from torch.nn.utils.rnn import pad_sequence

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


if __name__=="__main__":
    _,datalorder,_ = get_dataloader("./data/Dataset",20,4)
    for i,k in datalorder:
        print(len(i))
        break


