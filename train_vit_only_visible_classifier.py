import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, ConcatDataset
from torchvision import datasets, transforms, models
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# ===============================
# 📘 CONFIGURATION
# ===============================

device = torch.device("mps" if torch.backends.mps.is_available() else 
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🧠 Using device: {device}")

# Reuse your confirmed dataset paths
REAL_DATA_DIR = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/altogether"
SYNTHETIC_DATA_DIR = "/Users/ananyazabin/Research/iris/iris_gan_project/results/synthetic_visible_256_v4"

# Use same timestamped folder as before if available, otherwise create new
RESULTS_BASE = "/Users/ananyazabin/Research/iris/iris_gan_project/results"
existing_folders = sorted([f for f in os.listdir(RESULTS_BASE) if f.startswith("classifier_comparison_")])
RESULTS_DIR = os.path.join(RESULTS_BASE, existing_folders[-1]) if existing_folders else os.path.join(RESULTS_BASE, "classifier_comparison_vit_only")
os.makedirs(RESULTS_DIR, exist_ok=True)

# Parameters
BATCH_SIZE = 16
EPOCHS = 30
NUM_CLASSES = 5
SEED = 42
torch.manual_seed(SEED)

# ===============================
# 🧩 DATA PREPARATION (224×224 for ViT)
# ===============================

transform_vit = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

real_dataset = datasets.ImageFolder(root=REAL_DATA_DIR, transform=transform_vit)
synthetic_dataset = datasets.ImageFolder(root=SYNTHETIC_DATA_DIR, transform=transform_vit)

datasets_dict = {
    "real_only": real_dataset,
    "synthetic_only": synthetic_dataset,
    "real_plus_synthetic": ConcatDataset([real_dataset, synthetic_dataset])
}

# ===============================
# 🧠 MODEL: ViT-B16
# ===============================

def get_vit_model(num_classes):
    model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    return model.to(device)

# ===============================
# ⚙️ TRAINING FUNCTION
# ===============================

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, dataset_name):
    print(f"\n🚀 Training ViT-B16 on {dataset_name} dataset...")
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = torch.tensor(0.0, device=device)

            for inputs, labels in dataloaders[phase]:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data).float()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = (running_corrects / len(dataloaders[phase].dataset)).item()
            history[f"{phase}_loss"].append(epoch_loss)
            history[f"{phase}_acc"].append(epoch_acc)

            if phase == "val":
                scheduler.step(epoch_loss)

            print(f"Epoch [{epoch+1}/{num_epochs}] {phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if history['val_acc'][-1] > best_acc:
            best_acc = history['val_acc'][-1]
            torch.save(model.state_dict(), f"{RESULTS_DIR}/vit_b16_{dataset_name}_best.pth")

    # Save training plots
    plt.figure()
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f"Accuracy - ViT-B16 ({dataset_name})")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/vit_b16_{dataset_name}_accuracy.png")

    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f"Loss - ViT-B16 ({dataset_name})")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/vit_b16_{dataset_name}_loss.png")

    return model

# ===============================
# 🧮 RUN ONLY VIT-B16
# ===============================

all_results = []

for dataset_name, dataset in datasets_dict.items():
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len
    train_data, val_data, test_data = random_split(dataset, [train_len, val_len, test_len])

    dataloaders = {
        "train": DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True),
        "val": DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False),
        "test": DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    }

    model = get_vit_model(NUM_CLASSES)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

    trained_model = train_model(model, dataloaders, criterion, optimizer, scheduler, EPOCHS, dataset_name)

    # Evaluate
    trained_model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for inputs, labels in dataloaders['test']:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = trained_model(inputs)
            _, preds = torch.max(outputs, 1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(f"{RESULTS_DIR}/vit_b16_{dataset_name}_report.csv")

    acc = report['accuracy']
    macro_f1 = report['macro avg']['f1-score']
    all_results.append({"Model": "vit_b_16", "Dataset": dataset_name, "Accuracy": acc, "Macro_F1": macro_f1})

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=real_dataset.classes, yticklabels=real_dataset.classes)
    plt.title(f"Confusion Matrix - ViT-B16 ({dataset_name})")
    plt.savefig(f"{RESULTS_DIR}/vit_b16_{dataset_name}_confusion.png")
    plt.close()

# Save summary
summary_df = pd.DataFrame(all_results)
summary_csv = f"{RESULTS_DIR}/vit_only_summary.csv"
summary_df.to_csv(summary_csv, index=False)
print(f"\n✅ ViT-B16 training complete. Results saved to: {RESULTS_DIR}")
