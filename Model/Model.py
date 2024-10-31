import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import os

from tqdm import tqdm

from Dataset import Dataset
from Log import Loger

class Model(nn.Module):
    def __init__(self, name_model="Model", save=False, log: Loger = None, DEBUG=False):
        super(Model, self).__init__()

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
        
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        
        # Save model state, optimizer state, and epoch information
        torch.save({
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss
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
        epoch = checkpoint['epoch']
        loss = checkpoint['loss']
        
        self.log["INFO"](f"Checkpoint loaded from {checkpoint_path} at epoch {epoch} with best validation loss {loss:.4f}")

        return self


class ModelClassification(Model):
    def __init__(self, num_classes=1, name_model="ModelClassification", save=False, log: Loger = None, DEBUG=False):
        super(ModelClassification, self).__init__(name_model=name_model, save=save, log=log, DEBUG=DEBUG)

        self.num_classes = num_classes
        self.model = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes)
        )

        self.optimizer = optim.Adam(self.model.parameters())

    def forward(self, x):
        return self.model(x)
    
    def train(self, dataset: Dataset, batch_size=32, epochs=10, test: bool=False):
        if self.model is None:
            self.log["ERROR"]("Model is not initialized")
            raise Exception("Model is not initialized")
        
        loader = dataset.get_bath(batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)  # Scheduler
        running_loss = 0.0

        loop_train = tqdm(range(epochs), desc="Train", total=epochs, leave=False)

        for _ in loop_train:
            if test:
                bath_test = {}

            self.model.train()

            for bath in loader:
                for image, label in bath:
                    if test and len(bath_test.values()) <= len(dataset) * dataset.test_size and not image in bath_test:
                        bath_test[image] = label
                        continue

                    outputs = self.model(image)
                    loss = criterion(outputs, label)
                
                    self.optimizer.zero_grad()
                    loss.backward()
                    self.optimizer.step()

                    running_loss += loss.item()
            
            # self.log["INFO"](f"Epoch {self.epoch+1}/{epochs}, Loss: {running_loss/len(dataset)}")
            loop_train.set_description(f"Train | Epoch: {self.epoch+1}/{epochs} | Loss: {running_loss}",
                            refresh=True)
            
            self.epoch += 1

            if running_loss < self.best_loss:
                self.save_best(running_loss)
                self.best_loss = running_loss
        
        self.loss = running_loss / len(dataset)

        scheduler.step(self.loss)

        self.save_last()

        self.model.eval()

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