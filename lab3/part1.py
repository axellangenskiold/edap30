#%% 0. Imports

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models

# %% 1. Hyperparameters

BATCH_SIZE = 32
NUM_EPOCHS = 10
LR = 1e-3
NUM_CLASSES = 5
NUM_WORKERS = 4
SEED = 42

if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

torch.manual_seed(SEED)

#%% 2. Prepare the dataset

# resize -> center crop -> ImageNet normalization
weights = models.ResNet50_Weights.IMAGENET1K_V2
preprocess = weights.transforms()

train_ds = datasets.ImageFolder("flower-splits/train", transform=preprocess)
val_ds   = datasets.ImageFolder("flower-splits/val",   transform=preprocess)
test_ds  = datasets.ImageFolder("flower-splits/test",  transform=preprocess)

labels = train_ds.classes

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                      num_workers=NUM_WORKERS, pin_memory=True)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=True)

#%% 3. Model implementation

class FlowerNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

# %% 4. Prediction function

def predict_classes(model, x):
    # TODO: Implement
    pass

# %% 5. Evaluate

def evaluate(model, dl, labels):
    # TODO: Implement
    pass

# TODO: Test your evaluation over an unitialized model

# %% 6. Main train loop
def infinite_iter(dl):
    while True:
        for item in dl:
            yield item

# TODO: Implement your train function
def train(model, optimizer, train_dl, val_dl, num_epochs, name):
    # TODO: Implement
    pass

# %% 7. Train baseline
model = FlowerNet().to(device)

# TODO: Write training code and save your results

# %% 8. Evaluate

# TODO: Do full evaluation over all datasets

# %% 9. Attempt Image Augmentation, train for twice as long

# TODO: Write augmentation code, train and save your results.

# %% 10. Evaluate augmented model

# TOOD: Evalute on the test-set

# %% 11. Attempt another model for transfer-learning

# TODO: Implement your new model
    
# %% 12. Train your new model

# TODO: Reuse your train method and train on this new model

# %% 13. Evaluate the new model

# TODO: Evaluate your new model on the test dataset
