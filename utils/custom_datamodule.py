import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import lightning.pytorch as lp
import pytorch_lightning as pl
from pathlib import Path
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import timm.data

from utils.custom_dataset import PavingLawnDataset

class PavingLawnDatamodule(pl.LightningDataModule):
    def __init__(self,train_path,test_path,batch_size=8):
        super().__init__()
        self.train_path=train_path
        self.test_path=test_path
        self.train_dataset=None
        self.val_dataset=None
        self.test_dataset=None
        self.batch_size=batch_size
        self.augmentations = A.Compose([
            A.Resize(width=512, height=512),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()])

    def prepare_data(self):
        return super().prepare_data()

    

    def setup(self,stage):
        train_valid_dataset = PavingLawnDataset(self.train_path,self.augmentations)
        self.train_dataset,self.val_dataset = train_test_split(train_valid_dataset, test_size=0.2, random_state=42)
        self.test_dataset=PavingLawnDataset(self.test_path,self.augmentations)
        print('Datasets loaded')

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, num_workers=8)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=8)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=8)
