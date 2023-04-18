# %% SETUP AND IMPORTS
import torch
import os

import dataset
import model
import engine
import utils
from rich import print
from rich.pretty import pprint
from tqdm import tqdm

hardware = utils.runtime()
# Setup
print("> SETUP")
print("-----------------------------------------------")
print(f"Torch version: {torch.__version__}")
print(f"Device: {hardware}")
print()

# %% DATA LOADIING
dataloaders = dataset.get_cifar10(
    trdata = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trdata"),
    tsdata = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tsdata"),
    train_val_split=0.9,
    batch_size=128,
)

training, validation, testing = dataloaders
dataset.summary(training, validation, testing)
print()

# %% MODEL BUILDING
model_hyperparameters = {
    "input_shape": (3, 32, 32),
    "classes": 10, 
    "n_convs": 3, 
    "factor_conv": 3, 
    "n_fc": 4, 
    "factor_fc": 2, 
    "activation": "relu", 
    "dropout": 0.5,
    "accelerator": hardware
}
model = model.Model(model_hyperparameters).to(hardware)

print("> MODEL")
print("-----------------------------------------------")
model.summary()
print(model_hyperparameters)
model.check()
print()

# %% TRAINING
print("> TRAINING")
print("-----------------------------------------------")
training_hyperparameters = {
    "epochs":200,
    "optimizer":"Adam",
    "loss_fn":"CrossEntropyLoss",
    "learning_rate":1e-3,
    "weight_decay":1e-5,   
}
tr = engine.Training(model, training_hyperparameters)
print()
tr.train_loop(
    train_dataloader=training, 
    val_dataloader=validation
)
tr.save("outputs")
# summary(model, input_size=hyperparameters["input_shape"], device=device)
# %%
