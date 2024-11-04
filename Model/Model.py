import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms

from .HandlerDataset import HandlerDataset

import os

from tqdm import tqdm

from Dataset import Dataset
from Log import Loger

class Model(nn.Module):
    def __init__(self, name_model="Model", save=False, log: Loger = None, DEBUG=False):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = None
        self.optimizer = None
        
        self.name_model = f"{name_model}.pth" if not name_model.endswith(".pth") else name_model
        self._save = save
        self.epoch = 0
        self.loss = 0
        self.best_loss = float('inf')

        self.DEBUG = DEBUG
        if self.DEBUG:
            self.log = Loger()
        else:
            if log is None:
                log = Loger().off

            self.log = log
        
    def set_save(self, save=False):
        self._save = save
        self.log["INFO"](f"Save model: {self.save}")
    
    def save(self, checkpoint_path, message=""):
        if not self._save:
            return
        
        if self.model is None or self.optimizer is None:
            self.log["ERROR"]("Model or optimizer is not initialized")
            raise Exception("Model is not initialized") if self.model is None else Exception("Optimizer is not initialized")

        
        # Save model state, optimizer state, and epoch information
        torch.save({
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'best_loss': self.best_loss
        }, checkpoint_path)

        if message:
            self.log["INFO"](message)
        else:
            self.log["INFO"](f"Saved with loss {self.loss:.4f} at epoch {self.epoch+1} and checkpoint path: {checkpoint_path}")
        
    def save_best(self, best_loss, checkpoint_dir="checkpoints"):
        checkpoint_path = os.path.join(checkpoint_dir, f"best_{self.epoch + 1}_{self.name_model}")
        
        return self.save(checkpoint_path, 
                         message=f"Saved with best loss {best_loss:.4f} at epoch {self.epoch+1} and checkpoint path: {checkpoint_path}")

    def save_last(self, checkpoint_dir="checkpoints"):
        checkpoint_path = os.path.join(checkpoint_dir, f"last_{self.epoch + 1}_{self.name_model}")
        
        return self.save(checkpoint_path, 
                         message=f"Saved with loss {self.loss:.4f} at epoch {self.epoch+1} and checkpoint path: {checkpoint_path}")

    def load(self, checkpoint_path):
        """
        Loads the model and optimizer state from a checkpoint file.
        
        :param model: The model instance to load the weights into.
        :param optimizer: The optimizer instance to load the state into.
        :param checkpoint_path: Path to the checkpoint file.
        
        :return: epoch (int) - The epoch at which the checkpoint was saved.
                best_loss (float) - The best validation loss at checkpoint saving time.
        """
        if not os.path.exists(checkpoint_path):
            self.log["ERROR"]("Checkpoint path does not exist")
            raise Exception("Checkpoint path does not exist")
        
        checkpoint = torch.load(checkpoint_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.best_loss = checkpoint['best_loss']

        
        self.log["INFO"](f"Checkpoint loaded from {checkpoint_path} at epoch {self.epoch} with best validation loss {self.loss:.4f}")

        return self


class ModelClassification(Model):
    def __init__(self, num_classes=1, name_model="ModelClassification", save=False, log: Loger = None, DEBUG=False):
        
        super(ModelClassification, self).__init__(name_model=name_model, save=save, log=log, DEBUG=DEBUG)
        self.num_classes = num_classes

        self.layers = nn.ModuleList()

        self.add_conv_layer(3, 64)
        self.add_conv_layer(64, 128)
        self.add_conv_layer(128, 256)

        self.fc1 = None
        self.fc2 = nn.Linear(512, num_classes)

        self.optimizer = optim.Adam(self.parameters())

    def add_conv_layer(self, in_channels, out_channels, kernel_size=3):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        self.layers.append(conv_layer)
        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)

        x = nn.Flatten()(x)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 512)
        
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)
        x = self.fc2(x)

        return nn.Softmax(dim=1)(x)
    def train_in_dataset(self, dataset: Dataset, batch_size=32, epochs=10, test: bool=False):
        
        self.to(self.device)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        handler = HandlerDataset(dataset, transform=transform)
        loader = DataLoader(handler, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)  # Scheduler
        running_loss = 0.0

        criterion.to(self.device)

        for epoch in range(epochs):

            self.train()
            loop_train = tqdm(loader, desc=f"Train | Epoch: {epoch + 1}/{epochs}", leave=True)

            for images, labels in loop_train:
                # print(images, labels)
                # (images, labels) - (images.to(self.device), labels.to(self.device))
                outputs = self(images)
                loss = criterion(outputs, labels)
            
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item() / len(labels)
            
                # self.log["INFO"](f"Epoch {self.epoch+1}/{epochs}, Loss: {running_loss/len(dataset)}")
                loop_train.set_postfix(loss=running_loss)
            
            self.epoch += 1

            if running_loss < self.best_loss:
                self.save_best(running_loss)
                self.best_loss = running_loss
            
            scheduler.step(running_loss)
            loop_train.set_description(f"Train | Epoch: {epoch + 1}/{epochs} | Loss: {running_loss:.4f}", refresh=True)
            self.loss = running_loss
            running_loss = 0

        self.save_last()

        self.eval()

        return self


class TextFieldDetectorSSD(Model):
    def __init__(self, num_classes=2, name_model="ModelTextField", save=False, log: Loger = None, DEBUG=False):
        super(TextFieldDetectorSSD, self).__init__(name_model=name_model, save=save, log=log, DEBUG=DEBUG)
        self.num_classes = num_classes

        # Feature extraction layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)

        # Prediction layers for bounding boxes and class scores
        self.loc_head = nn.Conv2d(128, 4 * 4, kernel_size=3, padding=1)  # 4 bounding box coords per anchor
        self.cls_head = nn.Conv2d(128, num_classes * 4, kernel_size=3, padding=1)  # num_classes scores per anchor

        self.optimizer = optim.Adam(self.parameters())

    def forward(self, x):
        x = nn.ReLU()(self.conv1(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.ReLU()(self.conv2(x))
        x = nn.MaxPool2d(2, 2)(x)
        x = nn.ReLU()(self.conv3(x))
        
        loc_preds = self.loc_head(x).permute(0, 2, 3, 1).contiguous()
        cls_preds = self.cls_head(x).permute(0, 2, 3, 1).contiguous()

        # Reshape predictions to (batch, num_anchors, 4) for bbox and (batch, num_anchors, num_classes)
        loc_preds = loc_preds.view(loc_preds.size(0), -1, 4)
        cls_preds = cls_preds.view(cls_preds.size(0), -1, self.num_classes)

        return loc_preds, cls_preds
    
    def train(self, dataset: Dataset, batch_size=32, epochs=10, test: bool=False):
        if self.model is None:
            self.log["ERROR"]("Model is not initialized")
            raise Exception("Model is not initialized")
        
        train_loader = dataset.get_dataloader(batch_size=batch_size, shuffle=True)
        criterion_bbox = nn.MSELoss()  # Loss for bounding box regression
        criterion_class = nn.CrossEntropyLoss()  # Loss for classification
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)  # Scheduler
        running_loss = 0.0

        for _ in range(epochs):
            self.model.train()
            for images, labels in train_loader:
                self.optimizer.zero_grad()
                bboxes, label = labels
                pred_bboxes, pred_classes = self.model(images)
                loss_bbox = criterion_bbox(pred_bboxes, bboxes)
                loss_class = criterion_class(pred_classes, label)
                
                # Combined loss
                loss = loss_bbox + loss_class
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
            
            self.log["INFO"](f"Epoch {self.epoch+1}/{epochs}, Loss: {running_loss/len(train_loader)}")

            self.epoch += 1

            if running_loss < self.best_loss:
                self.save_best(running_loss)
                self.best_loss = running_loss
        
        self.loss = running_loss / len(train_loader)

        scheduler.step(self.loss)

        self.save_last(self.loss)

        return self