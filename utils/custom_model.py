
# Append project root
import os
import sys

import torchmetrics.metric
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch & Lightning
import torch
import torchmetrics
from torch.utils.data import random_split
import pytorch_lightning as pl
import segmentation_models_pytorch as smp

# Neptune
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

from variables import constans
from utils import common   
from variables import params
from variables import constans

# Garbage collector
import gc
gc.collect()


class LawnAndPaving(pl.LightningModule):
    def __init__(self, model, enc, classes, lr):
        super().__init__()
        self.enc=enc
        self.classes=classes
        self.learning_rate=lr
        self.accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=constans.NUM_CLASSES)
        self.loss_fn=smp.losses.DiceLoss(mode='multiclass', from_logits=True).to(common.device)
        self.save_hyperparameters()
        self.model = smp.create_model(arch=params.ARCH,
                            encoder_name=params.ENC_NAME,
                            encoder_weights=params.ENC_WEIGHTS,
                            in_channels=constans.IN_CHANNELS,
                            classes=constans.NUM_CLASSES).to(common.device)

    def optimizer_zero_grad(self, epoch, batch_idx, optimizer):
        optimizer.zero_grad(set_to_none=True)

    def forward(self, x):
        return self.model(x)
    
    def shared_step(self, batch, stage):
        inputs, labels = batch
        labels = labels.long()
        outputs = self.forward(inputs)
        labels = labels.unsqueeze(1)
        loss = self.loss_fn(outputs, labels)

        if torch.isinf(loss):
            return None
        self.accuracy.update(outputs, labels.squeeze())

        tp, fp, fn, tn = smp.metrics.get_stats(torch.argmax(outputs, 1).unsqueeze(1), labels.long(), mode='multiclass', num_classes = constans.NUM_CLASSES)

        iou = smp.metrics.iou_score(tp, fp, fn, tn, reduction="macro-imagewise")
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, sync_dist=True)
        self.log(f'{stage}_acc', self.accuracy, prog_bar=True)
        self.log(f'{stage}_iou', iou, prog_bar=True)

        return loss
    
    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")
    
    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        

    

