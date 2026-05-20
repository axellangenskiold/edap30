#%% 0. Imports
import torch
from torch import nn
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from torchvision.models import mobilenet_v3_large, MobileNet_V3_Large_Weights
from torchvision import transforms
import os
import json
from torchvision.models import resnet50, ResNet50_Weights


# %% 1. Hyperparameters
DATA_ROOT   = "flower-splits"
BATCH       = 32
NUM_WORKERS = 0
NUM_CLASSES=5
CLASS_NAMES= ['daisy', 'dandelion', 'rose', 'sunflower', 'tulip']
#print(os.getcwd())


#%% 2. Prepare the dataset


weights = MobileNet_V3_Large_Weights.IMAGENET1K_V2
eval_transform = weights.transforms()


train_transform = eval_transform


train_ds = ImageFolder(f"{DATA_ROOT}/train", transform=train_transform)
val_ds   = ImageFolder(f"{DATA_ROOT}/val",   transform=eval_transform)
test_ds  = ImageFolder(f"{DATA_ROOT}/test",  transform=eval_transform)
train_eval_ds = ImageFolder(f"{DATA_ROOT}/train", transform=eval_transform)

assert train_ds.classes == val_ds.classes == test_ds.classes == CLASS_NAMES, \
    f"Class mismatch: dataset has {train_ds.classes}"

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH*2,
                               shuffle=False, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH*2, shuffle=False,
                          num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH*2, shuffle=False,
                          num_workers=NUM_WORKERS)

#%% 3. Model implementation

class FlowerNet(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, freeze_backbone=True):
        super().__init__()
        weights= MobileNet_V3_Large_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = mobilenet_v3_large(weights=weights)

      
        in_features = self.backbone.classifier[3].in_features
        self.backbone.classifier[3] = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("classifier.3"):
                    p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)

# %% 4. Prediction function

def predict_classes(model, x):
    model.eval()
    with torch.no_grad():
        logits = model(x)
        preds = logits.argmax(dim=1)
    return preds

# %% 5. Evaluate

def evaluate(model, dl, labels):
    model.eval()

    all_preds, all_targets = [], []
    with torch.no_grad():
        for x, y in dl:
            logits = model(x)
            all_preds.append(logits.argmax(dim=1))
            all_targets.append(y)

    preds   = torch.cat(all_preds).numpy()
    targets = torch.cat(all_targets).numpy()

    acc         = (preds == targets).mean()
    macro_f1    = f1_score(targets, preds, average="macro")
    weighted_f1 = f1_score(targets, preds, average="weighted")

    print(f"Accuracy:    {acc:.4f}")
    print(f"Macro F1:    {macro_f1:.4f}")
    print(f"Weighted F1: {weighted_f1:.4f}\n")
    print(classification_report(targets, preds, target_names=labels, digits=4))

    return {
        "accuracy":    acc,
        "macro_f1":    macro_f1,
        "weighted_f1": weighted_f1,
        "preds":       preds,
        "targets":     targets,
        "confusion":   confusion_matrix(targets, preds),
    }

# TODO: Test your evaluation over an unitialized model

# %% 6. Main train loop


# TODO: Implement your train function
def train(model, optimizer, train_dl, val_dl, num_epochs, name):
    criterion = nn.CrossEntropyLoss()
    best_val_acc = 0.0
    history = {
        "epoch":        [],
        "train_loss":   [],
        "train_acc":    [],   
        "val_loss":     [],
        "val_acc":      [],
        "val_macro_f1": [],
    }

    for epoch in range(1, num_epochs + 1):

        model.train()
        running_loss, running_correct, n_samples = 0.0, 0, 0  
        for x, y in train_dl:
            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()
            running_loss    += loss.item() * x.size(0)
            running_correct += (logits.argmax(1) == y).sum().item()   
            n_samples       += x.size(0)
        train_loss = running_loss / n_samples
        train_acc  = running_correct / n_samples                       

    
        model.eval()
        val_running_loss, val_n = 0.0, 0
        all_preds, all_targets = [], []
        with torch.no_grad():
            for x, y in val_dl:
                logits = model(x)
                loss = criterion(logits, y)
                val_running_loss += loss.item() * x.size(0)
                val_n            += x.size(0)
                all_preds.append(logits.argmax(dim=1))
                all_targets.append(y)
        val_loss = val_running_loss / val_n
        preds   = torch.cat(all_preds).numpy()
        targets = torch.cat(all_targets).numpy()
        val_acc      = (preds == targets).mean()
        val_macro_f1 = f1_score(targets, preds, average="macro")

    
        history["epoch"].append(epoch)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(float(train_acc))               
        history["val_loss"].append(val_loss)
        history["val_acc"].append(float(val_acc))
        history["val_macro_f1"].append(float(val_macro_f1))


        marker = ""
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), f"{name}_best.pt")
            marker = " ★"

      
        print(f"Epoch {epoch:3d}/{num_epochs} | "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.4f} | "
              f"macro_f1={val_macro_f1:.4f}{marker}")

   
    with open(f"{name}_history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history

# %% 7. Train baseline
model = FlowerNet(num_classes=NUM_CLASSES, freeze_backbone=True)

optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3, weight_decay=1e-4,
)

history = train(
    model        = model,
    optimizer    = optimizer,
    train_dl     = train_loader,
    val_dl       = val_loader,
    num_epochs   = 5,
    name         = "baseline",
)

# TODO: Write training code and save your results

# %% 8. Evaluate


model.load_state_dict(torch.load("baseline_best.pt"))

print("=== TRAIN ===")
train_results = evaluate(model, train_eval_loader, labels=CLASS_NAMES)
print("\n=== VAL ===")
val_results   = evaluate(model, val_loader,        labels=CLASS_NAMES)
print("\n=== TEST ===")
test_results  = evaluate(model, test_loader,       labels=CLASS_NAMES)
# TODO: Do full evaluation over all datasets

# %% 9. Attempt Image Augmentation, train for twice as long
aug_train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.TrivialAugmentWide(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
    transforms.RandomErasing(p=0.25),
])


train_ds.transform = aug_train_transform

model_aug = FlowerNet(num_classes=NUM_CLASSES, freeze_backbone=True)
optimizer_aug = torch.optim.AdamW(
    [p for p in model_aug.parameters() if p.requires_grad],
    lr=1e-3, weight_decay=1e-4,
)

history_aug = train(
    model      = model_aug,
    optimizer  = optimizer_aug,
    train_dl   = train_loader,
    val_dl     = val_loader,
    num_epochs = 10,
    name       = "augmented",
)


# TODO: Write augmentation code, train and save your results.

# %% 10. Evaluate augmented model

model_aug.load_state_dict(torch.load("augmented_best.pt"))

print("\n=== TEST ===")
test_results  = evaluate(model_aug, test_loader,       labels=CLASS_NAMES)
# TOOD: Evalute on the test-set

# %% 11. Attempt another model for transfer-learning

weights = ResNet50_Weights.IMAGENET1K_V2
eval_transform = weights.transforms()

train_transform = eval_transform

train_ds = ImageFolder(f"{DATA_ROOT}/train", transform=train_transform)
val_ds   = ImageFolder(f"{DATA_ROOT}/val",   transform=eval_transform)
test_ds  = ImageFolder(f"{DATA_ROOT}/test",  transform=eval_transform)
train_eval_ds = ImageFolder(f"{DATA_ROOT}/train", transform=eval_transform)

assert train_ds.classes == val_ds.classes == test_ds.classes == CLASS_NAMES, \
    f"Class mismatch: dataset has {train_ds.classes}"

train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True,
                          num_workers=NUM_WORKERS, drop_last=True)
train_eval_loader = DataLoader(train_eval_ds, batch_size=BATCH*2,
                               shuffle=False, num_workers=NUM_WORKERS)
val_loader   = DataLoader(val_ds,   batch_size=BATCH*2, shuffle=False,
                          num_workers=NUM_WORKERS)
test_loader  = DataLoader(test_ds,  batch_size=BATCH*2, shuffle=False,
                          num_workers=NUM_WORKERS)

class FlowerNetNew(nn.Module):
    def __init__(self, num_classes=5, pretrained=True, freeze_backbone=True):
        super().__init__()
        weights= ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
        self.backbone = resnet50(weights=weights)

        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Linear(in_features, num_classes)

        if freeze_backbone:
            for name, p in self.backbone.named_parameters():
                if not name.startswith("fc"):
                    p.requires_grad = False

    def forward(self, x):
        return self.backbone(x)


# TODO: Implement your new model
    
# %% 12. Train your new model

model_new = FlowerNetNew(num_classes=NUM_CLASSES, freeze_backbone=True)
optimizer_new = torch.optim.AdamW(
    [p for p in model_new.parameters() if p.requires_grad],
    lr=1e-3, weight_decay=1e-4,
)

history_new = train(
    model      = model_new,
    optimizer  = optimizer_new,
    train_dl   = train_loader,
    val_dl     = val_loader,
    num_epochs = 5,
    name       = "new_pretrained",
)
# TODO: Reuse your train method and train on this new model

# %% 13. Evaluate the new model

model_new.load_state_dict(torch.load("new_pretrained_best.pt"))

print("\n=== TEST ===")
test_results  = evaluate(model_new, test_loader,       labels=CLASS_NAMES)
# TODO: Evaluate your new model on the test dataset

# %%
