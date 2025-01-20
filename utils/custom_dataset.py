import natsort
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variables import constans
from torch.utils.data import Dataset as BaseDataset
from PIL import Image
from matplotlib import pyplot as plt
from torchvision import transforms

class PavingLawnDataset(BaseDataset):
    def __init__(self,image_dir,classes,augmentations=None,mask_augmentations=None):
        self.image_dir = image_dir
        self.image_names = natsort.natsorted([f for f in os.listdir(image_dir) if f.lower().endswith('.jpg')])
        self.mask_names = natsort.natsorted([f for f in os.listdir(image_dir) if f.lower().endswith('.png')])

        self.classes = classes
        self.augmentations = augmentations
        self.mask_augmentations = mask_augmentations
        self.image_dir = image_dir

    def __len__(self):
        return len(self.image_names)
    
    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.image_names[idx])
        mask_path = os.path.join(self.image_dir, self.mask_names[idx])
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augmentations:
            image = self.augmentations(image)
        if self.mask_augmentations:
            mask = self.mask_augmentations(mask)

        return image, mask
    
    

if __name__ == "__main__":
    ds=PavingLawnDataset(constans.IMAGE_PATH,constans.CLASSES,None)
    
    for i in range(len(ds)):
        fig=plt.figure()
        image,mask = ds[i]
        plt.subplot(1, 2, 1)
        plt.title("Image")
        plt.imshow(image)
        plt.subplot(1, 2, 2)
        plt.title("Mask")
        plt.imshow(mask)
        plt.draw()
        plt.pause(0.001)
        plt.show()