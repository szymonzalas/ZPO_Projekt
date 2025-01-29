# Append project root
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Torch & Lightning
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint


# Neptune
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.loggers import NeptuneLogger

# From project folders
from utils import common
from utils.custom_model import LawnAndPaving
from utils.custom_datamodule import PavingLawnDatamodule
from variables import constans
from variables import params

# Garbage collector
import gc
gc.collect()

if __name__ == '__main__':
    data_module = PavingLawnDatamodule(
        train_path="data/train",
        test_path="data/test",
        batch_size=params.BATCH_SIZE
    )

    model = LawnAndPaving(
        params.ARCH,
        params.ENC_NAME,
        constans.NUM_CLASSES,
        params.LR
    )

    model_summary_callback = ModelSummary(max_depth=-1)

    neptune_logger = NeptuneLogger(
        project=common.neptune_project,
        api_token=common.neptune_key,
        tags=[params.ENC_NAME,params.ARCH,params.ENC_WEIGHTS,str(params.LR)],)
    
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=20,
        mode='min')
    
    checkpoint_callback = ModelCheckpoint(
        dirpath=f'checkpoint',
        filename=f'{params.ARCH}_{params.ENC_NAME}_{params.ENC_WEIGHTS}',
        monitor='val_loss',
        mode='min',
        save_top_k=1)
    
    trainer = Trainer(
        logger=neptune_logger,
        callbacks=[model_summary_callback, early_stop_callback, checkpoint_callback],
        accelerator='gpu',
        benchmark=True,
        max_epochs=params.MAX_EPOCH)
    
    trainer.fit(model, datamodule=data_module)
    trainer.test(model, data_module, ckpt_path='best')
    neptune_logger.finalize('DONE')