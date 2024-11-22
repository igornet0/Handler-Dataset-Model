import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.models as models

from .HandlerDataset import HandlerDataset

import os

from tqdm import tqdm

from Dataset import Dataset
from Log import Loger

class Model(nn.Module):

    name_model="Model"

    def __init__(self, num_classes=1, save=False, log: Loger = None, DEBUG=False):
        super(Model, self).__init__()

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.optimizer = None
        self.layers = []
        self.num_classes = num_classes

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

    def use_layers(self, x):

        for layer in self.layers:
            x = layer(x)

        return x

    def forward(self, x):
        return self.use_layers(x)

    def searh_model_file(self, path):
        if not os.path.exists(path):
            self.log["ERROR"]("Checkpoint path does not exist")
            return None

        if not os.path.isdir(path):
            if self.name_model in path:
                return path
            else:
                self.log["ERROR"]("Checkpoint path is not a directory")
                return None
            
        for file in os.listdir(path):
            if self.name_model in file:
                return os.path.join(path, file)

    def create_conv_layer(self, in_channels, out_channels, kernel_size=3):
        conv_layer = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )

        return conv_layer

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        
    def set_save(self, save=False):
        self._save = save
        self.log["INFO"](f"Save model: {self.save}")
    
    def save(self, checkpoint_path, message=""):
        if not self._save:
            return
        
        if self.model is None or self.optimizer is None:
            self.log["ERROR"]("Model or optimizer is not initialized")
            return None

        if not os.path.exists(os.path.dirname(checkpoint_path)):
            self.log["INFO"]("Create checkpoint directory")
            os.makedirs(os.path.dirname(checkpoint_path))
        
        # Save model state, optimizer state, and epoch information
        torch.save({
            'epoch': self.epoch + 1,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': self.loss,
            'best_loss': self.best_loss,
            'num_classes': self.num_classes
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
        Loads the model from the given checkpoint path.

        Args:
            checkpoint_path (str): path to the checkpoint file

        Returns:
            self: the model instance

        Raises:
            FileNotFoundError: if the checkpoint path does not exist
        """

        if not os.path.exists(checkpoint_path):
            self.log["ERROR"]("Checkpoint path does not exist")
            return None
        
        model_path = self.searh_model_file(checkpoint_path)

        checkpoint = torch.load(model_path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.loss = checkpoint['loss']
        self.best_loss = checkpoint['best_loss']
        self.num_classes = checkpoint['num_classes']

        
        self.log["INFO"](f"Checkpoint loaded from {checkpoint_path} at epoch {self.epoch} with best validation loss {self.loss:.4f}")

        return self
    
    @property
    def model(self):
        return self

    def train(self, loader, criterion, scheduler=None, epochs=10):

        if self.optimizer is None:
            self.log["ERROR"]("Optimizer is not initialized")
            return None

        running_loss = 0.0

        for epoch in range(epochs):

            self.model.train()
            loop_train = tqdm(loader, desc=f"Train | Epoch: {epoch + 1}/{epochs}", leave=True)

            for images, labels in loop_train:
                # print(images, labels)
                # (images, labels) - (images.to(self.device), labels.to(self.device))
                outputs = self.model(images)

                if outputs is None:
                    continue

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
            
            if scheduler:
                scheduler.step(running_loss)

            loop_train.set_description(f"Train | Epoch: {epoch + 1}/{epochs} | Loss: {running_loss:.4f}", refresh=True)
            self.loss += running_loss
            running_loss = 0

        self.loss = self.loss / epochs

        self.save_last()

        self.model.eval()

        return self


class ModelClassification(Model):

    name_model="ModelClassification"

    def __init__(self, num_classes=1, save=False, log: Loger = None, DEBUG=False):
        
        super(ModelClassification, self).__init__(num_classes=num_classes, save=save, log=log, DEBUG=DEBUG)

        self.layers = self.init_layers_conv()
        self.init_model()

    @classmethod
    def init_layers_conv(cls):
        layers = nn.ModuleList()

        layers.append(cls.create_conv_layer(3, 64))
        layers.append(cls.create_conv_layer(64, 128))
        layers.append(cls.create_conv_layer(128, 256))

        return layers

    def init_model(self):

        self.fc1 = None
        self.fc2 = nn.Linear(512, self.num_classes)

    def forward(self, x):

        x = self.use_layers(x)

        x = nn.Flatten()(x)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 512)
        
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout()(x)
        x = self.fc2(x)

        return nn.Softmax(dim=1)(x)
    
    def train_in_dataset(self, dataset: Dataset, batch_size=32, epochs=10, test: bool=False, learning_rate=0.001):
        
        self.model.to(self.device)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        handler = HandlerDataset(dataset, transform=transform)
        loader = DataLoader(handler, batch_size=batch_size)

        criterion = nn.CrossEntropyLoss()
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)  # Scheduler

        criterion.to(self.device)

        self.set_optimizer(optim.Adam(self.model.parameters(), lr=learning_rate))

        return super(ModelClassification, self).train(loader, criterion, scheduler, epochs)

class DocumentDetector(Model):

    name_model="DocumentDetector"

    def __init__(self):
        super(DocumentDetector, self).__init__()

        self.layers = self.init_layers_conv()
        self.init_model()

    def init_model(self):

        self.fc1 = None
        self.fc2 = nn.Linear(512, 4)

    @classmethod
    def init_layers_conv(cls):
        layers = nn.ModuleList()

        layers.append(cls.create_conv_layer(3, 64))
        layers.append(cls.create_conv_layer(64, 128))
        layers.append(cls.create_conv_layer(128, 256))

        return layers

    def forward(self, x):

        x = self.use_layers(x)

        x = nn.Flatten()(x)

        if self.fc1 is None:
            self.fc1 = nn.Linear(x.size(1), 512)
        
        x = self.fc1(x)
        x = nn.ReLU()(x)
        x = nn.Dropout(0.2)(x)
        x = self.fc2(x)

        return x

    def train_in_dataset(self, dataset: Dataset, batch_size=32, epochs=10, test: bool=False, learning_rate=0.001):
        
        self.model.to(self.device)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        handler = HandlerDataset(dataset, transform=transform)
        loader = DataLoader(handler, batch_size=batch_size)

        criterion = nn.SmoothL1Loss()
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)  # Scheduler

        criterion.to(self.device)

        self.set_optimizer(optim.Adam(self.model.parameters(), lr=learning_rate))

        return super(DocumentDetector, self).train(loader, criterion, scheduler, epochs)

class PolygonDetectorSSD(Model):

    name_model="PolygonDetectorSSD"

    def __init__(self):
        super(PolygonDetectorSSD, self).__init__()
        
        # Backbone: Pretrained ResNet50
        resnet = models.resnet50(pretrained=True)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-2])  # Remove classification layers

        # Additional Feature Layers for SSD
        self.extra_layers = nn.Sequential(
            nn.Conv2d(2048, 1024, kernel_size=3, stride=2, padding=1),  # Reduce spatial size
            nn.ReLU(),
            nn.Conv2d(1024, 512, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        # Predictors for bounding boxes and polygon vertices
        self.box_predictor = nn.Conv2d(512, 4 * 6, kernel_size=3, padding=1)  # Predict 6 bounding boxes per feature cell
        self.vertex_predictor = nn.Conv2d(512, 8 * 6, kernel_size=3, padding=1)  # Predict 8 vertices per box

    def convert_to_polygons(self, box_preds, vertex_preds, image_size):
        """
        Convert model outputs into absolute polygon coordinates.
        
        Args:
            box_preds (torch.Tensor): Bounding box predictions [batch_size, num_detections, 4].
            vertex_preds (torch.Tensor): Vertex predictions [batch_size, num_detections, 8].
            image_size (tuple): Size of the input image (width, height).
        
        Returns:
            polygons (list of list): Polygons for each detection as a list of (x, y) tuples.
        """
        polygons = []
        img_width, img_height = image_size

        for b in range(box_preds.size(0)):  # Iterate over batch
            for d in range(box_preds.size(1)):  # Iterate over detections
                # Extract bounding box and vertices
                cx, cy, w, h = box_preds[b, d]
                x1, y1, x2, y2, x3, y3, x4, y4 = vertex_preds[b, d]

                # Calculate bounding box corners
                x_min = (cx - w / 2) * img_width
                y_min = (cy - h / 2) * img_height
                x_max = (cx + w / 2) * img_width
                y_max = (cy + h / 2) * img_height

                # Convert vertices to absolute coordinates
                vertices = [
                    (x_min + x1 * w * img_width, y_min + y1 * h * img_height),
                    (x_min + x2 * w * img_width, y_min + y2 * h * img_height),
                    (x_min + x3 * w * img_width, y_min + y3 * h * img_height),
                    (x_min + x4 * w * img_width, y_min + y4 * h * img_height),
                ]
                polygons.append(vertices)

        return polygons

    def polygon_to_box_and_vertices(self,polygon):
        """
        Converts a polygon into a bounding box and normalized vertices.
        
        Args:
            polygon (list): List of 4 tuples representing polygon vertices [(x1, y1), ..., (x4, y4)].
        
        Returns:
            bbox (tuple): Bounding box (cx, cy, w, h).
            normalized_vertices (list): Normalized vertices [(nx1, ny1), ..., (nx4, ny4)].
        """
        # Extract x and y coordinates
        x_coords, y_coords = zip(*polygon)
        
        # Compute bounding box
        x_min, y_min = min(x_coords), min(y_coords)
        x_max, y_max = max(x_coords), max(y_coords)
        cx = (x_min + x_max) / 2
        cy = (y_min + y_max) / 2
        w = x_max - x_min
        h = y_max - y_min

        # Normalize vertices
        normalized_vertices = [
            ((x - x_min) / w, (y - y_min) / h) for x, y in polygon
        ]

        return (cx, cy, w, h), normalized_vertices

    def detection_loss(self, predictions, labels):
        pred_boxes, pred_vertices = predictions
        gt_boxes, gt_vertices = self.polygon_to_box_and_vertices(labels)

        box_loss = nn.SmoothL1Loss()(pred_boxes, gt_boxes)
        vertex_loss = nn.MSELoss()(pred_vertices, gt_vertices)
        return box_loss + vertex_loss

    def forward(self, x):
        # Extract features using backbone
        features = self.feature_extractor(x)

        # Process additional feature layers
        extra_features = self.extra_layers(features)

        # Predict bounding boxes and polygon vertices
        box_preds = self.box_predictor(extra_features)
        vertex_preds = self.vertex_predictor(extra_features)

        # Reshape predictions to match output format
        box_preds = box_preds.permute(0, 2, 3, 1).reshape(x.size(0), -1, 4)
        vertex_preds = vertex_preds.permute(0, 2, 3, 1).reshape(x.size(0), -1, 8)

        return box_preds, vertex_preds

    def train_in_dataset(self, dataset: Dataset, batch_size=32, epochs=10, test: bool=False, learning_rate=0.001):
        
        self.model.to(self.device)

        transform = transforms.Compose([
            transforms.ToTensor()
        ])

        handler = HandlerDataset(dataset, transform=transform)
        loader = DataLoader(handler, batch_size=batch_size)

        criterion = self.detection_loss
        scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=2, verbose=True)  # Scheduler

        criterion.to(self.device)

        self.set_optimizer(optim.Adam(self.model.parameters(), lr=learning_rate))

        return super(PolygonDetectorSSD, self).train(loader, criterion, scheduler, epochs)