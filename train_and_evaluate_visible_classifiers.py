try:
    import os
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, random_split, ConcatDataset
    from torchvision import datasets, transforms, models
    from sklearn.metrics import classification_report, confusion_matrix
    import seaborn as sns
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from datetime import datetime
except ModuleNotFoundError as e:
    print(f"\n❌ Missing library: {e.name}. Please install it using 'pip install {e.name}' before running this script.")
    raise SystemExit(1)

# ===============================
# 📘 CONFIGURATION
# ===============================

# Verify torch availability before proceeding
if not hasattr(torch, 'cuda') and not hasattr(torch, 'backends'):
    raise ImportError("Torch seems improperly installed or missing required modules. Please reinstall using 'pip install torch torchvision torchaudio'.")

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🧠 Using device: {device}")

REAL_DATA_DIR = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/altogether"
SYNTHETIC_DATA_DIR = "/Users/ananyazabin/Research/iris/iris_gan_project/results/synthetic_visible_256_v4"

TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
RESULTS_DIR = f"/Users/ananyazabin/Research/iris/iris_gan_project/results/classifier_comparison_{TIMESTAMP}"
os.makedirs(RESULTS_DIR, exist_ok=True)

BATCH_SIZE = 16
EPOCHS = 30
NUM_CLASSES = 5
SEED = 42
torch.manual_seed(SEED)

# ===============================
# 🧩 DATA PREPARATION
# ===============================
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

try:
    real_dataset = datasets.ImageFolder(root=REAL_DATA_DIR, transform=transform)
    synthetic_dataset = datasets.ImageFolder(root=SYNTHETIC_DATA_DIR, transform=transform)
except Exception as e:
    print(f"\n❌ Dataset loading failed: {e}\nCheck that your dataset paths are correct and accessible.")
    raise SystemExit(1)

datasets_dict = {
    "real_only": real_dataset,
    "synthetic_only": synthetic_dataset,
    "real_plus_synthetic": ConcatDataset([real_dataset, synthetic_dataset])
}

# ===============================
# 🧠 MODEL DEFINITIONS
# ===============================

def get_model(model_name, num_classes):
    if model_name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        model.heads.head = nn.Linear(model.heads.head.in_features, num_classes)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model.to(device)

# ===============================
# ⚙️ TRAINING AND EVALUATION FUNCTIONS
# ===============================

def train_model(model, dataloaders, criterion, optimizer, scheduler, num_epochs, model_name, dataset_name):
    print(f"\n🚀 Training {model_name.upper()} on {dataset_name} dataset...")
    best_acc = 0.0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss = 0.0
            running_corrects = torch.tensor(0.0, device=device, dtype=torch.float32)

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
            torch.save(model.state_dict(), f"{RESULTS_DIR}/{model_name}_{dataset_name}_best.pth")

    plt.figure()
    plt.plot(history['train_acc'], label='Train Acc')
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title(f"Accuracy - {model_name} ({dataset_name})")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_{dataset_name}_accuracy.png")

    plt.figure()
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.title(f"Loss - {model_name} ({dataset_name})")
    plt.legend()
    plt.savefig(f"{RESULTS_DIR}/{model_name}_{dataset_name}_loss.png")

    return model

# ===============================
# 🧮 FULL PIPELINE
# ===============================

all_results = []
models_to_train = ["resnet18", "efficientnet_b0", "vit_b_16"]

for model_name in models_to_train:
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

        model = get_model(model_name, NUM_CLASSES)
        criterion = nn.CrossEntropyLoss()

        if model_name == "vit_b_16":
            optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=5e-5)
        elif model_name == "efficientnet_b0":
            optimizer = optim.Adam(model.parameters(), lr=5e-4, weight_decay=1e-5)
        else:
            optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=3, factor=0.5)

        trained_model = train_model(model, dataloaders, criterion, optimizer, scheduler, EPOCHS, model_name, dataset_name)

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
        df_report.to_csv(f"{RESULTS_DIR}/{model_name}_{dataset_name}_report.csv")

        acc = report['accuracy']
        macro_f1 = report['macro avg']['f1-score']
        all_results.append({"Model": model_name, "Dataset": dataset_name, "Accuracy": acc, "Macro_F1": macro_f1})

        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=real_dataset.classes, yticklabels=real_dataset.classes)
        plt.title(f"Confusion Matrix - {model_name} ({dataset_name})")
        plt.savefig(f"{RESULTS_DIR}/{model_name}_{dataset_name}_confusion.png")
        plt.close()

summary_df = pd.DataFrame(all_results)
summary_csv = f"{RESULTS_DIR}/classifier_comparison_summary.csv"
summary_df.to_csv(summary_csv, index=False)

plt.figure(figsize=(8, 6))
sns.barplot(data=summary_df, x="Dataset", y="Accuracy", hue="Model", palette="viridis")
plt.title("Accuracy Comparison Across Models and Datasets")
plt.savefig(f"{RESULTS_DIR}/accuracy_comparison.png")

plt.figure(figsize=(8, 6))
sns.barplot(data=summary_df, x="Dataset", y="Macro_F1", hue="Model", palette="magma")
plt.title("Macro F1 Comparison Across Models and Datasets")
plt.savefig(f"{RESULTS_DIR}/f1_comparison.png")

print(f"\n✅ All results, models, and plots saved to: {RESULTS_DIR}")