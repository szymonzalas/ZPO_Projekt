ARCH = "unet" # ['unet', 'unetplusplus', 'manet', 'linknet', 'fpn', 'pspnet', 'deeplabv3', 'deeplabv3plus', 'pan', 'upernet', 'segformer']
ENC_NAME = "timm-resnest14d"
ENC_WEIGHTS = "imagenet"
LR = 1e-03
SEED=42
BATCH_SIZE=2

MAX_EPOCH=1000