import os
import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# --------------------------
# CONFIGURATION
# --------------------------
root_results = "/Users/ananyazabin/Research/iris/iris_gan_project/results/metrics_visible"
real_data_dir = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/train"
synthetic_data_dir = "/Users/ananyazabin/Research/iris/iris_gan_project/results/synthetic_visible_256_v4"

# 🔍 Automatically detect latest metrics_eval_* folder
metrics_folders = [f for f in os.listdir(root_results) if f.startswith("metrics_eval_")]
if not metrics_folders:
    raise FileNotFoundError("No metrics_eval_* folder found. Please run FID/SSIM evaluation first.")
latest_metrics = sorted(metrics_folders)[-1]
output_dir = os.path.join(root_results, latest_metrics)
print(f"📁 Using existing metrics folder for saving results:\n{output_dir}")

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# --------------------------
# HYPERPARAMETERS
# --------------------------
batch_size = 16
num_epochs = 15
learning_rate = 1e-4
val_split = 0.2
num_classes = 5

# --------------------------
# DATASET PREPARATION
# --------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_loaders(data_path):
    dataset = datasets.ImageFolder(data_path, transform=transform)
    val_len = int(len(dataset) * val_split)
    train_len = len(dataset) - val_len
    train_set, val_set = random_split(dataset, [train_len, val_len])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, dataset.classes

# --------------------------
# MODEL DEFINITION
# --------------------------
def get_resnet(num_classes):
    model = models.resnet50(weights="IMAGENET1K_V1")
    for param in model.parameters():
        param.requires_grad = False  # freeze base
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 512),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, num_classes)
    )
    return model.to(device)

# --------------------------
# TRAINING FUNCTION
# --------------------------
def train_model(model, train_loader, val_loader, criterion, optimizer):
    train_acc, val_acc, train_loss, val_loss = [], [], [], []
    for epoch in range(num_epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total
        train_loss.append(epoch_loss)
        train_acc.append(epoch_acc)

        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(device), labels.to(device)
                outputs = model(imgs)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)
        val_loss.append(val_running_loss / len(val_loader))
        val_acc.append(val_correct / val_total)

        print(f"Epoch [{epoch+1}/{num_epochs}] "
              f"Train Acc: {epoch_acc:.3f}, Val Acc: {val_acc[-1]:.3f}")

    return train_acc, val_acc, train_loss, val_loss

# --------------------------
# EVALUATION FUNCTION
# --------------------------
def evaluate_model(model, dataloader, class_names, label):
    model.eval()
    preds, targets = [], []
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs, 1)
            preds.extend(predicted.cpu().numpy())
            targets.extend(labels.numpy())

    acc = np.mean(np.array(preds) == np.array(targets))
    report = classification_report(targets, preds, target_names=class_names, output_dict=True)
    report_df = pd.DataFrame(report).transpose()

    # Confusion Matrix
    cm = confusion_matrix(targets, preds)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {label}")
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"confusion_matrix_{label}.png"))
    plt.close()

    return acc, report_df

# --------------------------
# EXPERIMENTS
# --------------------------
scenarios = {
    "real_only": real_data_dir,
    "real_plus_synthetic": [real_data_dir, synthetic_data_dir],
    "synthetic_only": synthetic_data_dir
}

final_results = []

for label, paths in scenarios.items():
    print(f"\n🧪 Running experiment: {label}")
    if isinstance(paths, list):
        merged_dataset_path = os.path.join(output_dir, f"merged_{label}")
        os.makedirs(merged_dataset_path, exist_ok=True)
        for src in paths:
            for cls in os.listdir(src):
                cls_src = os.path.join(src, cls)
                cls_dst = os.path.join(merged_dataset_path, cls)
                os.makedirs(cls_dst, exist_ok=True)
                for f in os.listdir(cls_src):
                    src_file = os.path.join(cls_src, f)
                    dst_file = os.path.join(cls_dst, f)
                    if not os.path.exists(dst_file):
                        os.system(f"cp '{src_file}' '{dst_file}'")
        data_path = merged_dataset_path
    else:
        data_path = paths

    train_loader, val_loader, class_names = get_loaders(data_path)
    model = get_resnet(len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.fc.parameters(), lr=learning_rate)

    train_acc, val_acc, train_loss, val_loss = train_model(model, train_loader, val_loader, criterion, optimizer)
    
    # Plot curves
    plt.figure()
    plt.plot(train_acc, label="Train Acc")
    plt.plot(val_acc, label="Val Acc")
    plt.legend()
    plt.title(f"Accuracy Curve - {label}")
    plt.savefig(os.path.join(output_dir, f"acc_curve_{label}.png"))
    plt.close()

    plt.figure()
    plt.plot(train_loss, label="Train Loss")
    plt.plot(val_loss, label="Val Loss")
    plt.legend()
    plt.title(f"Loss Curve - {label}")
    plt.savefig(os.path.join(output_dir, f"loss_curve_{label}.png"))
    plt.close()

    acc, report_df = evaluate_model(model, val_loader, class_names, label)
    report_df.to_csv(os.path.join(output_dir, f"classification_report_{label}.csv"))
    final_results.append({"Scenario": label, "Accuracy": acc})

# --------------------------
# SAVE SUMMARY
# --------------------------
results_df = pd.DataFrame(final_results)
results_df.to_csv(os.path.join(output_dir, "classification_summary.csv"), index=False)

print("\n🎯 Final Summary:")
print(results_df)
print(f"\n📂 All results saved to: {output_dir}")