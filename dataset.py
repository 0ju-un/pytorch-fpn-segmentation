import os
import numpy as np

from torch.utils.data import Dataset

class MyDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.transform = transform

    lst_data = os.listdir(self.data_dir)

    lst_label = [f for f in lst_data if f.startswith('label')]
    lst_input = [f for f in lst_data if f.startswith('input')]

    lst_label.sort()
    lst_input.sort()

    self.lst_label = lst_label
    self.lst_input = lst_input

  def __len__(self):
    return len(self.lst_input)

  def __getitem__(self, idx):
    label = np.zeros((3, 224, 224), dtype=np.float32)
    input = np.load(os.path.join(self.data_dir, self.lst_input[idx]))

    mask = np.load(os.path.join(self.data_dir, self.lst_label[idx]))
    for i in range(3):
      label[i][mask==i+1] = 1

    input = input.reshape(1, 224, 224).repeat(3, axis=0).transpose([1, 2, 0])

    if self.transform:
      input = self.transform(input)

    return [input, label]

class TestDataset(Dataset):
  def __init__(self, data_dir, transform=None):
    self.data_dir = data_dir
    self.transform = transform

    lst_data = os.listdir(self.data_dir)

    lst_input = [f for f in lst_data if f.startswith('input')]

    lst_input.sort()

    self.lst_input = lst_input

  def __len__(self):
    return len(self.lst_input)

  def __getitem__(self, idx):
    input = np.load(os.path.join(self.data_dir, self.lst_input[idx]))

    input = input.reshape(1, 224, 224).repeat(3, axis=0).transpose([1, 2, 0])

    if self.transform:
      input = self.transform(input)

    return input

