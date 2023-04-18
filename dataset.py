# %%
import torch
import torchvision
import os
from rich import print

def get_cifar10(trdata, tsdata, train_val_split, batch_size):

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        # Add transforms
    ])
    
    training_data = torchvision.datasets.CIFAR10(
        root=trdata,
        train = True,
        transform=transform,
    )
    
    testing_data = torchvision.datasets.CIFAR10(
        root=tsdata,
        train = False,
        transform=transform,
    )
    
    training, validation = torch.utils.data.random_split(
        training_data, 
        [train_val_split, 1-train_val_split]
    )
    
    testing = testing_data
    
    training_loader = torch.utils.data.DataLoader(training, batch_size=batch_size, shuffle=True)
    validation_loader = torch.utils.data.DataLoader(validation, batch_size=batch_size, shuffle=True)
    testing_loader = torch.utils.data.DataLoader(testing, batch_size=batch_size, shuffle=True)
    
    return training_loader, validation_loader, testing_loader    

def dataloader_summary(dataloader):
    print(f"Samples = {numel_samples(dataloader)}", end="\t")
    print(f"Batches = {numel_batches(dataloader)}")
    
def summary(training, validation, testing):
    print("> DATASET SUMMARY")
    print("-----------------------------------------------")
    print("TRAINING"); dataloader_summary(training)
    print("VALIDATION"); dataloader_summary(validation)
    print("TESTING"); dataloader_summary(testing)
    print()
    print(f"SHAPE: {list(get_shapes(training)[0])}")
    print(f"CLASSES: {len(get_classes(training)[0])}")
    
def numel_samples(dataloader):
    return len(dataloader.dataset)
    
def numel_batches(dataloader):
    return len(dataloader)

def sample(dataloader):
    return next(iter(dataloader))

def get_shapes(dataloader):
    return sample(dataloader)[0].shape[1:], sample(dataloader)[1].shape

def get_classes(dataloader):
    return sample(dataloader)[1].unique(return_counts=True)
    
    