
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
import torchmetrics
from torch.utils.data import DataLoader
from torch.utils.data import Dataset as BaseDataset
from torch.utils.data import random_split
from torch.optim import lr_scheduler
from torchvision import transforms
import pytorch_lightning as pl
import segmentation_models_pytorch as smp
from lightning.pytorch.tuner import Tuner

# Neptune
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

# From project folders
from utils import common
from utils.custom_dataset import PavingLawnDataset
from variables import constans
from variables import params

# Garbage collector
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
        self.accuracy=torchmetrics.Accuracy(num_classes=out_classes,task='multiclass')

    def forward(self, x):
        return self.model(x)

    def shared_step(self, batch, stage):
        image, mask = batch
        image = image.float()
        out = self.forward(image)
        loss = self.criterion(out, mask.long())
        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(out, 1).unsqueeze(1), mask.long(), mode='multiclass', num_classes=self.out_classes)
        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        self.log('train_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log('train_acc', self.accuracy, prog_bar=True)
        self.log('train_iou', iou, prog_bar=True)
        return {"loss": loss, "iou": iou}

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        result = self.shared_step(batch, "valid")
        self.log('valid_loss', result["loss"], on_step=True, on_epoch=True, prog_bar=True, sync_dist=True)
        return result

    def configure_optimizers(self):
        return self.optimizer
    

if __name__=="__main__":

    if common.neptune_log:
        neptune_logger = NeptuneLogger(
            project=common.neptune_project,
            api_token=common.neptune_key,
            tags=[params.ENC_NAME,params.ARCH,params.ENC_WEIGHTS,str(params.LR)],)
        
        early_stop_callback = EarlyStopping(
        monitor='valid_loss',
        min_delta=0.00,
        patience=10,
        verbose=True,
        mode='min')

        model_summary_callback = pl.callbacks.ModelSummary(max_depth=-1)
        
        cbs = pl.callbacks.ModelCheckpoint(
        dirpath=f'./checkpoints_{common.params["arch"]}',
        filename=common.params["arch"],
        verbose=True,
        monitor='valid_loss',
        mode='min',
        save_top_k=5)

    model = smp.create_model(arch=common.params['arch'],
                            encoder_name=common.params['enc_name'],
                            encoder_weights=common.params['enc_weights'],
                            in_channels=common.params['in_channels'],
                            classes=constans.NUM_CLASSES).to(common.device)

    optimizer = torch.optim.Adam(model.parameters(), lr=common.params['lr'])
    criterion = smp.losses.DiceLoss(mode='multiclass', from_logits=True).to(common.device)

    cbs = pl.callbacks.ModelCheckpoint(dirpath=f'./checkpoints/{common.params["arch"]}',
                                    filename=common.params["arch"],
                                    verbose=True,
                                    monitor='valid_loss',
                                    mode='min')

    pl_model = LawnAndPaving(model, common.params['arch'], common.params['num_classes'], criterion, optimizer)
    if common.neptune_log:
        trainer = pl.Trainer(logger=neptune_logger,callbacks=[early_stop_callback, cbs,model_summary_callback], accelerator='gpu', max_epochs=common.params['max_epoch'])
    else:
        trainer = pl.Trainer(callbacks=[cbs,model_summary_callback], accelerator='gpu', max_epochs=common.params['max_epoch'])

    tuner = tuner.Tuner()
    trainer.fit(pl_model, common.train_loader, common.valid_loader)


    # model = smp.create_model(arch,
    #                         encoder_name = enc_name,
    #                         encoder_weights = "imagenet",
    #                         in_channels = 3,
    #                         classes = classes).to(device)

    # state_dict = torch.load(cbs.best_model_path)['state_dict']
    # pl_state_dict = OrderedDict([(key[6:], state_dict[key]) for key in state_dict.keys()])

    # model.load_state_dict(pl_state_dict)
    # model.eval()
    # with torch.no_grad(): 

    #     outputs = []
    #     test_loss = 0.0
    #     iou = 0

    #     for image, mask in tqdm(test_loader):       
    #         image=image.float()
    #         image = image.to(device); mask = mask.to(device)
    #         output = model(image).to(device)
    #         tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(output, 1).unsqueeze(1), mask.long(), mode='multiclass', num_classes = 5)
    #         outputs.append({"tp": tp, "fp": fp, "fn": fn, "tn": tn})
    #         loss = criterion(output, mask.long()) 
    #         test_loss += loss.item() 
        
    #     tp = torch.cat([x["tp"] for x in outputs])
    #     fp = torch.cat([x["fp"] for x in outputs])
    #     fn = torch.cat([x["fn"] for x in outputs])
    #     tn = torch.cat([x["tn"] for x in outputs])
        
    #     print(f'Test Loss: {test_loss / len(test_loader)}')
    #     print('IoU:', smp.metrics.iou_score(tp, fp, fn, tn, reduction="micro-imagewise").item())

    #     random.seed(10)
    #     samples = random.sample(range(len(test_dataset)), 3)

    #     palette = [[155,38,182], [14,135,204], [124,252,0]]
    #     pal = [value for color in palette for value in color]

    #     cols = ['Image', 'Mask', 'Prediction']
    #     fig, axes = plt.subplots(len(samples), 3, figsize=(60, 40), sharex='row', sharey='row', 
    #                             subplot_kw={'xticks':[], 'yticks':[]}, tight_layout=True)

    #     for ax, col in zip(axes[0], cols): ax.set_title(col, fontsize=20) # set column label --> considered epoch
            
    #     for i in range(len(samples)):
    #         image, mask = test_dataset[samples[i]]
    #         image = image.float()
    #         pred = torch.argmax(model(torch.tensor(image).unsqueeze(0).to(device)), 1)

    #         mask = Image.fromarray(mask.squeeze(0).cpu().numpy()).convert('P')
    #         pred = Image.fromarray(np.array(pred.squeeze(0).cpu()).astype('uint8')).convert('P')
    #         mask.putpalette(pal)
    #         pred.putpalette(pal)

    #         axes[i, 0].imshow(np.array(image).transpose(1, 2, 0))
    #         axes[i, 1].imshow(mask)
    #         axes[i, 2].imshow(pred)
                
    #     fig.savefig(arch + '.png')