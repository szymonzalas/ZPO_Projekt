import natsort
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from variables import constans
from utils import common
from utils.custom_model import LawnAndPaving
from utils.custom_dataset import PavingLawnDataset
from variables import constans
from variables import params

import segmentation_models_pytorch as smp
import torch


checkpoint_path = 'checkpoint/unet_efficientnet-b0_imagenet.ckpt'
model = LawnAndPaving.load_from_checkpoint(checkpoint_path)
model.eval()
model.to('cuda')

x = torch.rand(1, 3, 512, 512).to('cuda')
_ = model(x)
export_model_name = f"{params.ARCH}_{params.ENC_NAME}_{params.ENC_WEIGHTS}.onnx"
torch.onnx.export(model,
                  x,
                  export_model_name,
                  export_params=True,
                  opset_version=15,
                  input_names=['input'],
                  output_names=['output'],
                  do_constant_folding=False)