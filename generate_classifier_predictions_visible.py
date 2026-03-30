import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report
import pandas as pd
import numpy as np
from tqdm import tqdm
import datetime

# -------------------------------
# Configuration
# -------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# Folders
base_data = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/test"
base_results = "/Users/ananyazabin/Research/iris/iris_gan_project/results/metrics_visible"

# Model paths
model_real = os.path.join(base_results, "classifier_real_only.pth")
model_synth = os.path.join(base_results, "classifier_synthetic_only.pth")
model_comb = os.path.join(base_results, "classifier_real_plus_synthetic.pth")

# Output folder
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(base_results, f"predictions_eval_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# -------------------------------
# Dataset and Transform
# -------------------------------
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

test_dataset = datasets.ImageFolder(root=base_data, transform=transform)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
class_names = test_dataset.classes
print(f"📁 Test set loaded — {len(test_dataset)} samples across {len(class_names)} classes.")

# -------------------------------
# Model Loader
# -------------------------------
def load_classifier(model_path):
    """Load a fine-tuned ResNet-based classifier."""
    model = models.resnet18(pretrained=False)
    model.fc = nn.Linear(model.fc.in_features, len(class_names))
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    return model

# -------------------------------
# Evaluation Function
# -------------------------------
def evaluate_model(model, loader, model_name):
    """Evaluate classifier, return predictions & metrics."""
    y_true, y_pred = [], []

    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc=f"Evaluating {model_name}"):
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            preds = outputs.argmax(dim=1)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(preds.cpu().numpy())

    # Convert to labels
    y_true_labels = [class_names[i] for i in y_true]
    y_pred_labels = [class_names[i] for i in y_pred]

    # Classification report
    report = classification_report(
        y_true_labels, y_pred_labels, target_names=class_names, output_dict=True
    )
    df_report = pd.DataFrame(report).transpose()
    df_report.to_csv(os.path.join(output_dir, f"classification_report_{model_name}.csv"))

    # Prediction CSV
    df_pred = pd.DataFrame({
        "y_true": y_true_labels,
        "y_pred": y_pred_labels
    })
    df_pred.to_csv(os.path.join(output_dir, f"predictions_{model_name}.csv"), index=False)

    print(f"✅ Saved predictions and report for {model_name}")
    return report["accuracy"], df_report

# -------------------------------
# Run Evaluations
# -------------------------------
models_to_test = {
    "real_only": model_real,
    "synthetic_only": model_synth,
    "real_plus_synthetic": model_comb
}

results = []

for name, path in models_to_test.items():
    if not os.path.exists(path):
        print(f"⚠️ Skipping {name} — model file not found: {path}")
        continue
    model = load_classifier(path)
    acc, report_df = evaluate_model(model, test_loader, name)
    results.append({"Model": name, "Accuracy": acc})

# -------------------------------
# Summary Save
# -------------------------------
summary = pd.DataFrame(results)
summary.to_csv(os.path.join(output_dir, "classification_predictions_summary.csv"), index=False)
print(f"\n📊 Summary saved to: {output_dir}/classification_predictions_summary.csv")

print("\n✅ All prediction CSVs and reports saved successfully.")