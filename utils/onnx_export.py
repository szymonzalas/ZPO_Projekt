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
import onnx
import json


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

modelo = onnx.load(export_model_name)

class_names = {
    0: '_background',
    1: 'lawn',
    2: 'paving',
}

m1 = modelo.metadata_props.add()
m1.key = 'model_type'
m1.value = json.dumps('Segmentor')

m2 = modelo.metadata_props.add()
m2.key = 'class_names'
m2.value = json.dumps(class_names)

m3 = modelo.metadata_props.add()
m3.key = 'resolution'
m3.value = json.dumps(50)

# optional, if you want to standarize input after normalisation
m4 = modelo.metadata_props.add()
m4.key = 'standardization_mean'
m4.value = json.dumps([0.0, 0.0, 0.0])

m5 = modelo.metadata_props.add()
m5.key = 'standardization_std'
m5.value = json.dumps([1.0, 1.0, 1.0])

onnx.save(modelo, export_model_name)