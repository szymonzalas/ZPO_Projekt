import natsort
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variables import constans
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms
import numpy as np

class PavingLawnDataset(BaseDataset):
    def __init__(self,image_dir,augmentations,debug_msg=False):
        self.image_dir = image_dir
        self.augmentations = augmentations
        print(image_dir)
        self.image_names = natsort.natsorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
        self.mask_names = natsort.natsorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])
        if debug_msg:
            print(f"Found {len(self.image_names)} images in {self.image_dir}")
            print(f"Found {len(self.mask_names)} masks in {self.image_dir}")

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.image_dir, self.mask_names[idx])
        image = np.asarray(Image.open(image_path).convert('RGB'))
        print(image)
        mask = np.asarray(Image.open(mask_path))
        transformed = self.augmentations(image=image, mask=mask)
        return transformed['image'], transformed['mask']