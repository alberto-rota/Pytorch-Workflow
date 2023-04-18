
# %%
import torch.nn as nn
import torch
import os
import shutil
from rich import print
from rich.table import Table
from rich.console import Console
from torchsummary import summary

class Model(nn.Module):

    def __init__(self, hyperparameters):
        self.hyperparameters = hyperparameters
        self.input_shape = hyperparameters["input_shape"]
        self.classes = hyperparameters["classes"]
        self.n_convs = hyperparameters["n_convs"]
        self.factor_conv = hyperparameters["factor_conv"]
        self.n_fc = hyperparameters["n_fc"]
        self.factor_fc = hyperparameters["factor_fc"]
        self.activation = hyperparameters["activation"]
        self.dropout = hyperparameters["dropout"]
        self.accelerator = hyperparameters["accelerator"]
        
        act = {"relu": nn.ReLU(), "sigmoid": nn.Sigmoid()}

        super(Model, self).__init__()
        shape = self.input_shape

        self.fe_layers = []
        self.shapes = [shape]
        self.cl_layers = []

        for c in range(self.n_convs):
            self.fe_layers.append(
                nn.Conv2d(
                    in_channels=shape[0],
                    out_channels=shape[0]*self.factor_conv,
                    kernel_size=3,
                    padding="same",
                )
            )
            shape = (shape[0]*self.factor_conv, shape[1], shape[2])
            self.shapes.append(shape)

            self.fe_layers.append(nn.ReLU())
            self.shapes.append(shape)

            self.fe_layers.append(nn.MaxPool2d(2))
            shape = (shape[0], shape[1]//2, shape[2]//2)
            self.shapes.append(shape)

        self.cl_layers.append(nn.Flatten())
        shape = (shape[0]*shape[1]*shape[2],)
        self.shapes.append(shape)

        for c in range(self.n_fc-1):
            self.cl_layers.append(
                nn.Linear(
                    in_features=shape[0],
                    out_features=shape[0]//self.factor_fc,
                )
            )
            shape = (shape[0]//self.factor_fc, )
            self.shapes.append(shape)

            assert self.activation in act.keys(
            ), f"Activation function not supported, choose from: {act.keys}"
            self.cl_layers.append(act[self.activation])
            self.shapes.append(shape)

            self.cl_layers.append(nn.Dropout(self.dropout))
            self.shapes.append(shape)

        self.cl_layers.append(
            nn.Linear(
                in_features=shape[0],
                out_features=self.classes,
            )
        )
        shape = (self.classes, )
        self.shapes.append(shape)

        self.cl_layers.append(nn.Softmax(dim=1))
        self.shapes.append(shape)

        self.featureextractor = nn.Sequential(*self.fe_layers)
        self.classifier = nn.Sequential(*self.cl_layers)

    def forward(self, x, return_features=False):
        features = self.featureextractor(x)
        prediction = self.classifier(features)
        if not return_features:
            return prediction
        else:
            return prediction, features

    def check(self):
        input_data = torch.rand(1, *self.input_shape).to(self.accelerator)
        try:
            out = self.forward(input_data)
        except Exception as e:
            print("Model doesn't output")
            print(f"Exception raised: <<< [red]{e}[/red] >>> ")
            return False
        if not out.size() == (1, self.classes):
            print(f"Output shape is not the expected one. {out.size()} != {(1, self.classes)}")
            return False
        return True

    def summary(self):
        console = Console()
        table = Table(show_header=True)
        table.add_column("Input Shape")
        table.add_column("Layer ")
        table.add_column("Output Shape")
        for i in range(len(self.fe_layers)):
            table.add_row(
                str(self.shapes[i]),
                str(self.fe_layers[i]),
                str(self.shapes[i+1])
            )
        for i in range(len(self.cl_layers)):
            table.add_row(
                str(self.shapes[i+len(self.fe_layers)]),
                str(self.cl_layers[i]),
                str(self.shapes[i+1+len(self.fe_layers)])
            )
        console.print(table)
    
# device ="cuda" if torch.cuda.is_available() else "cpu"

# hyperparameters = {
#     "input_shape": (3, 32, 32),
#     "classes": 10, 
#     "n_convs": 4, 
#     "factor_conv": 3, 
#     "n_fc": 4, 
#     "factor_fc": 2, 
#     "activation": "relu", 
#     "dropout": 0.2
# }

# print(f"Using device",device)
# model = Model(hyperparameters).to(device)
# summary(model, input_size=hyperparameters["input_shape"], device=device)