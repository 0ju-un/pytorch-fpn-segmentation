import argparse

import os

import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, datasets, models

from dataset import MyDataset
from FPN import FPN
from trainer import Trainer

## Parser 생성
parser = argparse.ArgumentParser()

parser.add_argument("--lr", default=1e-3, type=float, dest="lr")
parser.add_argument("--batch_size", default=64, type=int, dest="batch_size")
parser.add_argument("--num_epoch", default=100, type=int, dest="num_epoch")
parser.add_argument("--num_workers", default=0, type=int, dest="num_workers")
parser.add_argument("--n_class", default=3, type=int, dest="n_class")

parser.add_argument("--root_dir", default="", type=str, dest="root_dir")

parser.add_argument("--mode", default="FPN", type=str, dest="mode")
parser.add_argument("--load_checkpoint", default="", type=str, dest="load_checkpoint")

args = parser.parse_args()

## 트레이닝 파라메터 설정
n_class = 3

lr = args.lr
batch_size = args.batch_size
num_epoch = args.num_epoch
num_workers = args.num_workers

root_dir = args.root_dir
data_dir = os.path.join(root_dir, 'dataset')
ckpt_dir = os.path.join(root_dir, 'ckpt_dir')

mode = args.mode
backbone = "resnet50"
load_checkpoint = args.load_checkpoint

## 디렉터리 생성
if not os.path.exists(ckpt_dir):
    os.mkdir(ckpt_dir)

## 데이터셋 생성
trans = transforms.Compose([
  transforms.ToTensor(),
  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]) # imagenet
])

train_set = MyDataset(os.path.join(data_dir, "train"), transform=trans)
val_set = MyDataset(os.path.join(data_dir, "val"), transform=trans)

dataloaders = {
  'train': DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0),
  'val': DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=0)
}

## 네트워크 생성
if mode=="FPN":
    model = FPN(encoder_name=backbone,
                decoder_pyramid_channels=256,
                decoder_segmentation_channels=128,
                classes=n_class,
                dropout=0.3,
                activation='softmax',
                final_upsampling=4,
                decoder_merge_policy='add')## Optimizer 설정

## 네트워크 학습
model_trainer = Trainer(model=model, dataloaders=dataloaders, optimizer=optim.Adam,
                        lr=lr, batch_size=batch_size, num_epochs=num_epoch,
                        model_path=ckpt_dir, load_checkpoint=load_checkpoint)
model_trainer.start()