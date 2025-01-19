import os
import sys
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.optim import lr_scheduler
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from dotenv import load_dotenv, dotenv_values 
from torch.utils.data import random_split
from torchvision import transforms
from collections import OrderedDict
from tqdm import tqdm
import random
from PIL import Image

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from lightning_modules.custom_dataset import PavingLawnDataset
from variables import constans

import gc
gc.collect()

class LawnAndPaving(pl.LightningModule):
    def __init__(self, model, arch, out_classes, criterion, optimizer):
        super().__init__()
        self.model = model
        self.arch = arch
        self.out_classes = out_classes
        self.criterion = criterion
        self.optimizer = optimizer

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        image, mask = batch
        image = image.float()
        out = self.forward(image)
        loss = self.criterion(out, mask.long())
        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(out, 1).unsqueeze(1), mask.long(), mode='multiclass', num_classes=self.out_classes)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        self.log(f"{stage}_IoU", iou, prog_bar=True, on_epoch=True)
        self.log(f"{stage}_loss", loss)
        return {"loss": loss, "iou": iou}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "valid")

    def configure_optimizers(self):
        return self.optimizer
    

if __name__=="__main__":
    load_dotenv()

    if not torch.cuda.is_available():
        device=torch.device("cpu")
        print("Current device:", device)
    else:
        device=torch.device("cuda")
        print("Current device:", device, "- Type:", torch.cuda.get_device_name(0))

    transform = transforms.Compose([
                transforms.PILToTensor()
                ])
    dataset=PavingLawnDataset(constans.IMAGE_PATH,constans.CLASSES,transform,transform)

    train_dataset, valid_dataset, test_dataset  = random_split(dataset, [0.7, 0.2, 0.1])

    print("Total number of samples in the dataset:", len(dataset))
    print("Number of samples in the training set:", len(train_dataset))
    print("Number of samples in the validation set:", len(valid_dataset))
    print("Number of samples in the test set:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=4)
    valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=4)

    arch = 'unet'
    enc_name = 'efficientnet-b0'
    classes = 5

    model = smp.create_model(arch,
                            encoder_name=enc_name,
                            encoder_weights="imagenet",
                            in_channels=3,
                            classes=classes).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-03)
    criterion = smp.losses.DiceLoss(mode='multiclass', from_logits=True).to(device)
    cbs = pl.callbacks.ModelCheckpoint(dirpath=f'./checkpoints_{arch}',
                                    filename=arch,
                                    verbose=True,
                                    monitor='valid_loss',
                                    mode='min')

    pl_model = LawnAndPaving(model, arch, classes, criterion, optimizer)
    trainer = pl.Trainer(callbacks=cbs, accelerator='gpu', max_epochs=2)
    trainer.fit(pl_model, train_loader, valid_loader)


    model = smp.create_model(arch,
                            encoder_name = enc_name,
                            encoder_weights = "imagenet",
                            in_channels = 3,
                            classes = classes).to(device)

    state_dict = torch.load(cbs.best_model_path)['state_dict']
    pl_state_dict = OrderedDict([(key[6:], state_dict[key]) for key in state_dict.keys()])

    model.load_state_dict(pl_state_dict)
    model.eval()
    with torch.no_grad(): 

        outputs = []
        test_loss = 0.0
        iou = 0

        for image, mask in tqdm(test_loader):       
            image=image.float()
            image = image.to(device); mask = mask.to(device)
            output = model(image).to(device)
            tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(output, 1).unsqueeze(1), mask.long(), mode='multiclass', num_classes = 5)
            outputs.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
            loss = criterion(output, mask.long()) 
            test_loss += loss.item() 
        
        tp = torch.cat([x["tp"] for x in outputs])
        fp = torch.cat([x["fp"] for x in outputs])
        fn = torch.cat([x["fn"] for x in outputs])
        tn = torch.cat([x["tn"] for x in outputs])
        
        print(f'Test Loss: {test_loss / len(test_loader)}')
        print('IoU:', smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item())

        random.seed(10)
        samples = random.sample(range(len(test_dataset)), 3)

        palette = [[155,38,182], [14,135,204], [124,252,0]]
        pal = [value for color in palette for value in color]

        cols = ['Image', 'Mask', 'Prediction']
        fig, axes = plt.subplots(len(samples), 3, figsize=(60, 40), sharex='row', sharey='row', 
                                subplot_kw={'xticks':[], 'yticks':[]}, tight_layout=True)

        for ax, col in zip(axes[0], cols): ax.set_title(col, fontsize=20) # set column label --> considered epoch
            
        for i in range(len(samples)):
            image, mask = test_dataset[samples[i]]
            image = image.float()
            pred = torch.argmax(model(torch.tensor(image).unsqueeze(0).to(device)), 1)

            mask = Image.fromarray(mask.squeeze(0).cpu().numpy()).convert('P')
            pred = Image.fromarray(np.array(pred.squeeze(0).cpu()).astype('uint8')).convert('P')
            mask.putpalette(pal)
            pred.putpalette(pal)

            axes[i, 0].imshow(np.array(image).transpose(1, 2, 0))
            axes[i, 1].imshow(mask)
            axes[i, 2].imshow(pred)
                
        fig.savefig(arch + '.png')