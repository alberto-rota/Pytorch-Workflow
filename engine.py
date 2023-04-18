
import torch
import torch.nn as nn
import json, os
from tqdm import tqdm
import time
from rich import print
from torch.utils.tensorboard import SummaryWriter

class Training:
    
    def __init__(self,
                 model: nn.Module, 
                 hyperparameters: dict):
        
        # One training has one model
        self.model = model
        self.hyperparameters = hyperparameters
        # Epochs and batch size
        self.epochs = hyperparameters["epochs"]
        # self.batch_size = hyperparameters["batch_size"]
        
        # Optimizer prameter
        self.momentum = hyperparameters.get("momentum",0)
        self.learning_rate = hyperparameters["learning_rate"]
        self.weight_decay = hyperparameters.get("weight_decay",0)
        
        # Device and accelerator
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Timetable: logs total training time, average epoch training and inference time,
        # average batch training and inference time,
        self.chrono = {
            "total_time": 0,
            "epoch_training_time": 0, 
            "batch_training_time": 0,
            "batch_inference_time": 0,
            "sample_inference_time": 0,
        }
        
        optimizer_dict = {
            "adam": torch.optim.Adam(
                params = self.model.parameters(),
                lr = self.learning_rate,
                weight_decay=self.weight_decay,
            ),
            "sgd": torch.optim.SGD(
                params = self.model.parameters(),
                lr = self.learning_rate,
                momentum=self.momentum,
            ),
            # Add more optimizers
        }
        
        loss_fn_dict = {
            "crossentropyloss": nn.CrossEntropyLoss(),
            # Add more loss functions 
        }
        self.optimizer = optimizer_dict[hyperparameters["optimizer"].lower()]
        self.loss_fn = loss_fn_dict[hyperparameters["loss_fn"].lower()]
        
        self.metrics = {
            "training_loss": [],
            "training_accuracy": [],
            "validation_loss": [],
            "validation_accuracy": [],
        }

        self.chrono = {
            "epoch_training_time": 0,
            "batch_training_time": 0,
            "epoch_inference_time": 0,
            "batch_inference_time": 0,
        }
        
        TB_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tboard")
        
        tbruns = os.listdir(TB_DIR)
        runidx = int(tbruns[-1].split("run")[-1])+1 if len(tbruns)>0 else 0
        print(f"Training instantiated: Logging at /tboard/run{runidx}")
        os.mkdir(os.path.join(TB_DIR,f"run{runidx}"))

        self.tboard = SummaryWriter(log_dir=os.path.join(TB_DIR,f"run{runidx}"))
        self.runidx = runidx
        self.tboard.add_graph(self.model, torch.rand(1,*self.model.input_shape).to(self.device))
        
    # Executes dynamic training routines - LRScheduler, EarlyStopping, etc.
    def adapt(self):
        pass
    
    def tboard_update(self):
        pass
    
    def train_epoch(self, epoch, dataloader : torch.utils.data.DataLoader ):
        
        self.model.train()
        train_loss, train_acc = 0, 0
        for batch, (image, label) in enumerate(dataloader):
            
            self.optimizer.zero_grad()
            # Adapts tensors to hardware accelerator
            image,label = image.to(self.device), label.to(self.device)
            
            prediction_oh = self.model(image).to(self.device)
            loss = self.loss_fn(prediction_oh, label)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            
            prediction = prediction_oh.argmax(dim=1)
            train_acc += (prediction == label).sum().item()/len(dataloader)
            
            print(f"E {epoch+1}/{self.epochs} | B {batch+1}/{len(dataloader)} | T_Loss={loss.item():.4f} - T_Acc={(train_acc):.4f}%", end="\r")
        print()
            
        train_loss = train_loss / len(dataloader)
        
        self.metrics["training_loss"].append(train_loss)
        self.metrics["training_accuracy"].append(train_acc)
        
        return train_loss, train_acc
    
    
    def validate_epoch(self, epoch, dataloader : torch.utils.data.DataLoader ):
        
        self.model.eval()
        val_loss, val_acc = 0, 0
        with torch.no_grad():
            for batch, (image, label) in enumerate(dataloader):
                image,label = image.to(self.device), label.to(self.device)
                prediction_oh = self.model(image)
                loss = self.loss_fn(prediction_oh, label)
                val_loss += loss.item()
                prediction = prediction_oh.argmax(dim=1)
                val_acc += ((prediction == label).sum().item()/len(prediction))
                
        val_loss = val_loss / len(dataloader)
        val_acc = val_acc / len(dataloader)
        
        self.metrics["validation_loss"].append(val_loss)
        self.metrics["validation_accuracy"].append(val_acc)
        
        return val_loss, val_acc
    
    def print_epoch(self, epoch):
        for m in self.metrics.keys():
            print(f"{m}={self.metrics[m][-1]:.4f} - ",end="")   
        print()
         
    def tboard_log(self, STEP):
        for m in self.metrics.keys():
            self.tboard.add_scalar(m, self.metrics[m][-1], STEP)
        
    def train_loop(self, train_dataloader, val_dataloader):
        for e in range(self.epochs):
            print(f"Epoch {e}/{self.epochs} ")
            self.train_epoch(e, train_dataloader)
            self.validate_epoch(e, val_dataloader)
            self.print_epoch(e)
            self.tboard_log(e)
            
    def save(self,path):
        
        SAVE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), path,"run"+str(self.runidx))
        os.mkdir(os.path.join(SAVE_DIR))
        torch.save(self.model.state_dict(), os.path.join(SAVE_DIR, "model_state.pt"))
        torch.save(self.model, os.path.join(SAVE_DIR, "model.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(SAVE_DIR, "optimizer_state.pt"))
        
        with open(os.path.join(SAVE_DIR, "modelHP.json"), "w") as f:
            json.dump(self.model.hyperparameters, f, indent=1)
        with open(os.path.join(SAVE_DIR, "trainingHP.json"), "w") as f:
            json.dump(self.hyperparameters, f, indent=1)
            
        