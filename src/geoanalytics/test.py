# %%
from lightning import Trainer

from data.module import CustomDataModule, CustomDataConfig
from model.interface import BulkTransformerConfig, BulkTransformerInterface
import data.config as config


# %%

data_config = CustomDataConfig(
    config.HEXES_DATA_PATH,
    config.HEXES_TARGET_PATH,
    config.BANK_DATA_PATH,
    config.TRAINING_DATA_DIRECTORY,
    config.BANK_TARGET_PATH,
    config.CATEGORICAL_FEATURES,
    config.CONTINUOUS_FEATURES,
    fraction=config.FRACTION,
    seed=0,
)

datamodule = CustomDataModule(data_config, config.BATCH_SIZE)

model_config = BulkTransformerConfig(
    16,
    config.HEADS,
    config.DROPOUT,
    config.LAYERS,
    config.FEEDFORWARD_SIZE,
    config.CONTINUOUS_FEATURES,
    config.CATEGORICAL_FEATURES,
    config.CATEGORICAL_SIZES,
    datamodule.get_target_hexes(),
)

model = BulkTransformerInterface(model_config, config.LEARNING_RATE)
# %%
sum(parameter.numel() for parameter in model.model.transformer.parameters())

# %%
trainer = Trainer(devices=1, max_epochs=config.EPOCHS)
trainer.fit(model, datamodule)

# %%
datamodule.setup("predict")
model.predict_step(next(iter(datamodule.predict_dataloader())))
