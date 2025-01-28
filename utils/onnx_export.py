import natsort
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variables import constans
from utils import common
from main import LawnAndPaving
from utils.custom_dataset import PavingLawnDataset

import segmentation_models_pytorch as smp
import torch


model = smp.create_model(arch=common.params['arch'],
                        encoder_name=common.params['enc_name'],
                        encoder_weights=common.params['enc_weights'],
                        in_channels=common.params['in_channels'],
                        classes=constans.NUM_CLASSES).to(common.device)

optimizer = torch.optim.Adam(model.parameters(), lr=common.params['lr'])
criterion = smp.losses.DiceLoss(mode='multiclass', from_logits=True).to(common.device)

lawn_model = LawnAndPaving.load_from_checkpoint(
    "checkpoints/unet/unet-v1.ckpt",
    model=model,
    arch="Unet",
    out_classes=3,
    criterion=criterion,
    optimizer=optimizer
)
lawn_model.eval()
x = next(iter(common.train_loader))[0].float().to(common.device)

model_to_export = lawn_model.model

torch.onnx.export(
    model_to_export,
    x[:1],
    'model.onnx',
    export_params=True,
    opset_version=15,
    input_names=['input'],
    output_names=['output'],
    do_constant_folding=True,
    dynamic_axes={
        'input': {0: 'batch_size'},
        'output': {0: 'batch_size'}
    }
)
