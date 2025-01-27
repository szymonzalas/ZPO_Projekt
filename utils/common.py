
# Append project root
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Standard library
import random
from dotenv import load_dotenv, dotenv_values 

# Basic data manipulation and pyplot
import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from collections import OrderedDict

# Torch & Lightning
import torch
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import random_split
from torch.optim import lr_scheduler
from torchvision import transforms
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

# From project folders
from utils.custom_dataset import PavingLawnDataset
from variables import constans
from variables import params

# CODE STARTS HERE
# Load environment variables for neptune logger
load_dotenv()
neptune_project=os.getenv("neptune_project")
neptune_key=os.getenv("neptune_key")
neptune_log=False

if neptune_project==None or neptune_key==None:
    print("Neptune environment variables not found.")
else:
    neptune_log=True
    print("Neptune environment variables loaded.")


# Check for CUDA devices
if not torch.cuda.is_available():
    device=torch.device("cpu")
    print("Current device:", device)
else:
    device=torch.device("cuda")
    print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))


# Define transformations
transform_train = transforms.Compose([
                  transforms.PILToTensor()
                  ])


transform_valid = transforms.Compose([
                  transforms.PILToTensor()
                  ])

transform_test  = transforms.Compose([
                  transforms.PILToTensor()
                  ])

# RNG Generator
generator = torch.Generator().manual_seed(params.SEED)

# Load datasets
#train_dataset = PavingLawnDataset(constans.TRAIN_PATH, constans.CLASSES, transform_train,transform_train)
#valid_dataset = PavingLawnDataset(constans.VALID_PATH, constans.CLASSES, transform_valid,transform_valid)
train_valid_dataset = PavingLawnDataset(constans.TRAIN_PATH, constans.CLASSES, transform_train,transform_train)
train_dataset,valid_dataset=random_split(train_valid_dataset, [400, 50],generator=generator)
test_dataset = PavingLawnDataset(constans.TEST_PATH, constans.CLASSES, transform_test,transform_test)

# Load dataloaders
train_loader = DataLoader(train_dataset, batch_size=constans.BATCH_SIZE, shuffle=True, num_workers=constans.NUM_WORKERS)
valid_loader = DataLoader(valid_dataset, batch_size=constans.BATCH_SIZE, shuffle=False, num_workers=constans.NUM_WORKERS)
test_loader = DataLoader(test_dataset, batch_size=constans.BATCH_SIZE, shuffle=False, num_workers=constans.NUM_WORKERS)

# Model parameters
params={
    "num_classes" : constans.NUM_CLASSES,
    "batch_size" : constans.BATCH_SIZE,
    "num_workers" : constans.NUM_WORKERS,
    "num_classes" : constans.NUM_CLASSES,
    "in_channels": constans.IN_CHANNELS,
    "arch":params.ARCH,
    "enc_name" : params.ENC_NAME,
    "enc_weights" : params.ENC_WEIGHTS,
    "max_epoch" : params.MAX_EPOCH,
    "lr" : params.LR,
    "seed":params.SEED
    }



print("Common loaded")


# Validation and printing if run as main
if __name__=="__main__":
    UNSAFE_KEY_PRINT=False
    print("Parameters:")
    for p in params:
        print("- ",p,":",params[p])

    print("Number of samples in the training set:", len(train_dataset))
    print("Number of samples in the validation set:", len(valid_dataset))
    print("Number of samples in the test set:", len(test_dataset))

    print("Neptune logging: ",neptune_log)
    print("Neptune logger project:",neptune_project)
    print("Neptune logger key len:",len(neptune_key)) # For security reasons, only length is printed, expect ~150 characters
    if UNSAFE_KEY_PRINT:
        print("Neptune logger key:",neptune_key)
    