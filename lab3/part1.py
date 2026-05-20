#%% 0. Imports

import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.metrics import (
        ConfusionMatrixDisplay,
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score
    )
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms

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
                      num_workers=NUM_WORKERS, pin_memory=False,
                      persistent_workers=NUM_WORKERS > 0)
val_dl   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=False,
                      persistent_workers=NUM_WORKERS > 0)
test_dl  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                      num_workers=NUM_WORKERS, pin_memory=False,
                      persistent_workers=NUM_WORKERS > 0)

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
    model.eval()
    with torch.no_grad():
        logits = model(x.to(device))
        return logits.argmax(dim=1)

# %% 5. Evaluate

def evaluate(model, dl, labels):
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")

    all_preds, all_targets = [], []
    total_loss, total_count = 0.0, 0

    with torch.no_grad():
        for x, y in dl:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            total_loss += criterion(logits, y).item()
            total_count += y.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_targets.append(y.cpu())

    y_pred = torch.cat(all_preds).numpy()
    y_true = torch.cat(all_targets).numpy()

    return {
        "loss":     total_loss / total_count,
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro"),
        "cm":       confusion_matrix(y_true, y_pred),
        "report":   classification_report(y_true, y_pred,
                                          target_names=labels,
                                          digits=4, zero_division=0),
    }


# Sanity check: evaluate runs end-to-end on a freshly-instantiated model.
# Pretrained backbone + random head -> predictions should be ~chance (~20%).
# Guarded so DataLoader workers don't re-run training when they re-import this file.
if __name__ == "__main__":
    _sanity = FlowerNet().to(device)
    _metrics = evaluate(_sanity, val_dl, labels)
    print(f"[sanity] untrained val: acc={_metrics['accuracy']:.4f} "
          f"macro_f1={_metrics['macro_f1']:.4f} loss={_metrics['loss']:.4f}")
    del _sanity

# %% 6. Main train loop
def infinite_iter(dl):
    while True:
        for item in dl:
            yield item

def train(model, optimizer, train_dl, val_dl, num_epochs, name):
    criterion = nn.CrossEntropyLoss()
    out_dir = Path("runs") / name
    out_dir.mkdir(parents=True, exist_ok=True)

    history = {"train_loss": [], "train_acc": [],
               "val_loss":   [], "val_acc":   [], "val_macro_f1": []}
    best_val_f1 = -1.0

    for epoch in range(1, num_epochs + 1):
        model.train()
        running_loss, running_correct, running_total = 0.0, 0, 0

        for x, y in train_dl:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            logits = model(x)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            running_loss    += loss.item() * y.size(0)
            running_correct += (logits.argmax(dim=1) == y).sum().item()
            running_total   += y.size(0)

        train_loss = running_loss    / running_total
        train_acc  = running_correct / running_total
        val = evaluate(model, val_dl, labels)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val["loss"])
        history["val_acc"].append(val["accuracy"])
        history["val_macro_f1"].append(val["macro_f1"])

        print(f"[{name}] epoch {epoch}/{num_epochs}  "
              f"train: loss={train_loss:.4f} acc={train_acc:.4f}  |  "
              f"val: loss={val['loss']:.4f} acc={val['accuracy']:.4f} "
              f"f1={val['macro_f1']:.4f}")

        if val["macro_f1"] > best_val_f1:
            best_val_f1 = val["macro_f1"]
            torch.save(model.state_dict(), out_dir / "best.pt")

    torch.save(model.state_dict(), out_dir / "last.pt")
    with open(out_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    return history

# %% 7. Train baseline
if __name__ == "__main__":
    model = FlowerNet().to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    baseline_history = train(model, optimizer, train_dl, val_dl,
                             num_epochs=NUM_EPOCHS, name="baseline")

# %% 8. Evaluate

def plot_history(history, name):
    """Save train/val loss + accuracy curves to runs/<name>/curves.png."""
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, (ax_loss, ax_acc) = plt.subplots(1, 2, figsize=(12, 4))

    ax_loss.plot(epochs, history["train_loss"], label="train")
    ax_loss.plot(epochs, history["val_loss"],   label="val")
    ax_loss.set(xlabel="epoch", ylabel="loss", title=f"{name} - loss")
    ax_loss.legend()

    ax_acc.plot(epochs, history["train_acc"], label="train")
    ax_acc.plot(epochs, history["val_acc"],   label="val")
    ax_acc.set(xlabel="epoch", ylabel="accuracy", title=f"{name} - accuracy")
    ax_acc.legend()

    fig.tight_layout()
    fig.savefig(Path("runs") / name / "curves.png", dpi=120, bbox_inches="tight")
    plt.show()

def full_eval(model, name, dls=None):
    """Report loss/accuracy/macro-F1 + classification report on all splits."""
    if dls is None:
        dls = [("train", train_dl), ("val", val_dl), ("test", test_dl)]
    for split_name, dl in dls:
        m = evaluate(model, dl, labels)
        print(f"\n=== {name} / {split_name} ===")
        print(f"loss={m['loss']:.4f}  acc={m['accuracy']:.4f}  "
              f"macro_f1={m['macro_f1']:.4f}")
        print(m["report"])
        disp = ConfusionMatrixDisplay(confusion_matrix=m["cm"],
                                      display_labels=labels)
        disp.plot(cmap="Blues", values_format="d", xticks_rotation=45)
        plt.title(f"{name} / {split_name}")
        plt.tight_layout()
        plt.show()

# Reload the best checkpoint (highest val macro-F1) for reporting.
if __name__ == "__main__":
    model.load_state_dict(torch.load("runs/baseline/best.pt", map_location=device))
    plot_history(baseline_history, "baseline")
    full_eval(model, "baseline")

# %% 9. Attempt Image Augmentation, train for twice as long

# Reuse the matched preprocessing's crop size + normalization so the augmented
# pipeline ends in the exact same value range the pretrained backbone expects.
train_aug_transform = transforms.Compose([
    transforms.RandomResizedCrop(preprocess.crop_size[0], scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=preprocess.mean, std=preprocess.std),
])

train_aug_ds = datasets.ImageFolder("flower-splits/train",
                                    transform=train_aug_transform)
train_aug_dl = DataLoader(train_aug_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=False,
                          persistent_workers=NUM_WORKERS > 0)

if __name__ == "__main__":
    model_aug = FlowerNet().to(device)
    optimizer_aug = torch.optim.Adam(model_aug.parameters(), lr=LR)
    aug_history = train(model_aug, optimizer_aug, train_aug_dl, val_dl,
                        num_epochs=NUM_EPOCHS * 2, name="aug")

# %% 10. Evaluate augmented model

if __name__ == "__main__":
    model_aug.load_state_dict(torch.load("runs/aug/best.pt", map_location=device))
    plot_history(aug_history, "aug")
    full_eval(model_aug, "aug")

# %% 11. Attempt another model for transfer-learning

# Swap the backbone family entirely: EfficientNet-B0 (~5M params, mobile-inverted
# bottlenecks) vs ResNet50 (~25M params, residual conv blocks). Same ImageNet
# pretraining, but its weights ship with their own resize/crop and normalization
# stats, so we build matched train/eval transforms for it.
class FlowerNetEffNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES):
        super().__init__()
        self.backbone = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1
        )
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.backbone(x)

effnet_weights    = models.EfficientNet_B0_Weights.IMAGENET1K_V1
effnet_preprocess = effnet_weights.transforms()

train_effnet_transform = transforms.Compose([
    transforms.RandomResizedCrop(effnet_preprocess.crop_size[0], scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.RandomRotation(15),
    transforms.ToTensor(),
    transforms.Normalize(mean=effnet_preprocess.mean, std=effnet_preprocess.std),
])

train_effnet_ds = datasets.ImageFolder("flower-splits/train",
                                       transform=train_effnet_transform)
val_effnet_ds   = datasets.ImageFolder("flower-splits/val",
                                       transform=effnet_preprocess)
test_effnet_ds  = datasets.ImageFolder("flower-splits/test",
                                       transform=effnet_preprocess)

train_effnet_dl = DataLoader(train_effnet_ds, batch_size=BATCH_SIZE, shuffle=True,
                             num_workers=NUM_WORKERS, pin_memory=False,
                             persistent_workers=NUM_WORKERS > 0)
val_effnet_dl   = DataLoader(val_effnet_ds,   batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False,
                             persistent_workers=NUM_WORKERS > 0)
test_effnet_dl  = DataLoader(test_effnet_ds,  batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=NUM_WORKERS, pin_memory=False,
                             persistent_workers=NUM_WORKERS > 0)

# %% 12. Train your new model

if __name__ == "__main__":
    model_effnet     = FlowerNetEffNet().to(device)
    optimizer_effnet = torch.optim.Adam(model_effnet.parameters(), lr=LR)
    effnet_history   = train(model_effnet, optimizer_effnet,
                             train_effnet_dl, val_effnet_dl,
                             num_epochs=NUM_EPOCHS * 2, name="effnet")

# %% 13. Evaluate the new model

if __name__ == "__main__":
    model_effnet.load_state_dict(torch.load("runs/effnet/best.pt",
                                            map_location=device))
    plot_history(effnet_history, "effnet")
    full_eval(model_effnet, "effnet",
              dls=[("train", train_effnet_dl),
                   ("val",   val_effnet_dl),
                   ("test",  test_effnet_dl)])
