import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder

# --------------------------------------------------
# 🔍 1. Automatically find the latest results folder
# --------------------------------------------------
base_dir = "/Users/ananyazabin/Research/iris/iris_gan_project/results"
pattern = os.path.join(base_dir, "**", "classification_summary.csv")

summary_files = glob.glob(pattern, recursive=True)
if not summary_files:
    raise FileNotFoundError("No 'classification_summary.csv' found in results directory!")

# Get the most recently modified summary file
summary_file = max(summary_files, key=os.path.getmtime)
result_dir = os.path.dirname(summary_file)

# Infer other CSV paths
real_path = os.path.join(result_dir, "classification_report_real_only.csv")
synthetic_path = os.path.join(result_dir, "classification_report_synthetic_only.csv")
combined_path = os.path.join(result_dir, "classification_report_real_plus_synthetic.csv")

# Optionally look for predictions (if classifier stored them)
real_pred_path = os.path.join(result_dir, "predictions_real_only.csv")
synthetic_pred_path = os.path.join(result_dir, "predictions_synthetic_only.csv")
combined_pred_path = os.path.join(result_dir, "predictions_real_plus_synthetic.csv")

print(f"📂 Using files from: {result_dir}")
print(f"  ├─ Summary: {os.path.basename(summary_file)}")
print(f"  ├─ Real-only report: {os.path.basename(real_path)}")
print(f"  ├─ Synthetic-only report: {os.path.basename(synthetic_path)}")
print(f"  └─ Combined report: {os.path.basename(combined_path)}")

# --------------------------------------------------
# 🧾 2. Load CSVs safely
# --------------------------------------------------
def safe_read_csv(file_path):
    """Try reading as comma, tab, or whitespace-separated automatically."""
    try:
        df = pd.read_csv(file_path)
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, sep="\t")
        if df.shape[1] == 1:
            df = pd.read_csv(file_path, delim_whitespace=True)
        return df
    except Exception as e:
        raise ValueError(f"Error reading {file_path}: {e}")

summary = safe_read_csv(summary_file)
real = safe_read_csv(real_path)
synthetic = safe_read_csv(synthetic_path)
combined = safe_read_csv(combined_path)

print("\n✅ Summary columns detected:", list(summary.columns))

# --------------------------------------------------
# 🧮 3. Calculate Macro F1 automatically
# --------------------------------------------------
macro_f1_real = real["f1-score"].mean()
macro_f1_synthetic = synthetic["f1-score"].mean()
macro_f1_combined = combined["f1-score"].mean()

if "Scenario" in summary.columns:
    summary = summary.rename(columns={"Scenario": "Dataset"})

summary["Macro F1"] = [macro_f1_real, macro_f1_combined, macro_f1_synthetic]

# --------------------------------------------------
# 📁 4. Create output directory
# --------------------------------------------------
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(base_dir, f"classifier_analysis_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

# --------------------------------------------------
# 📊 5. Accuracy Comparison
# --------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x="Dataset", y="Accuracy", hue="Dataset", dodge=False, palette="viridis", legend=False)
plt.title("Classification Accuracy Comparison")
plt.ylabel("Accuracy")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{output_dir}/accuracy_comparison.png")
plt.close()

# --------------------------------------------------
# 📈 6. Macro F1 Comparison
# --------------------------------------------------
plt.figure(figsize=(8, 5))
sns.barplot(data=summary, x="Dataset", y="Macro F1", hue="Dataset", dodge=False, palette="plasma", legend=False)
plt.title("Macro F1 Score Comparison")
plt.ylabel("F1 Score")
plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig(f"{output_dir}/f1_comparison.png")
plt.close()

# --------------------------------------------------
# 🧩 7. Per-class F1 Comparison
# --------------------------------------------------
def prepare_metrics(df, label):
    """Ensure 'class' column exists and reshape metrics for plotting."""
    if "class" not in df.columns:
        df = df.reset_index().rename(columns={"index": "class"})
    df["Dataset"] = label
    return df.melt(
        id_vars=["class", "Dataset"],
        value_vars=["precision", "recall", "f1-score"],
        var_name="Metric",
        value_name="Value"
    )

real_melt = prepare_metrics(real, "Real")
synthetic_melt = prepare_metrics(synthetic, "Synthetic")
combined_melt = prepare_metrics(combined, "Combined")

merged = pd.concat([real_melt, synthetic_melt, combined_melt], axis=0)

plt.figure(figsize=(10, 6))
sns.barplot(
    data=merged[merged["Metric"] == "f1-score"],
    x="class",
    y="Value",
    hue="Dataset",
    palette="Set2"
)
plt.title("Per-Class F1 Score Comparison")
plt.ylabel("F1 Score")
plt.xlabel("Class")
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(f"{output_dir}/per_class_f1_comparison.png")
plt.close()

# --------------------------------------------------
# 🔲 8. Confusion Matrix Heatmaps (if available)
# --------------------------------------------------
def plot_confusion_matrix(csv_path, title, save_path):
    if not os.path.exists(csv_path):
        print(f"⚠️  No predictions file found for {title}. Skipping confusion matrix.")
        return
    preds = pd.read_csv(csv_path)
    if not {"y_true", "y_pred"}.issubset(preds.columns):
        print(f"⚠️  File {csv_path} must contain 'y_true' and 'y_pred' columns.")
        return

    le = LabelEncoder()
    labels = sorted(list(set(preds["y_true"]) | set(preds["y_pred"])))
    cm = confusion_matrix(preds["y_true"], preds["y_pred"], labels=labels)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    plt.figure(figsize=(6, 5))
    sns.heatmap(cm_norm, annot=True, fmt=".2f", cmap="Blues", xticklabels=labels, yticklabels=labels)
    plt.title(f"Confusion Matrix — {title}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"✅ Saved confusion matrix: {save_path}")

plot_confusion_matrix(real_pred_path, "Real-only Model", f"{output_dir}/confusion_matrix_real.png")
plot_confusion_matrix(synthetic_pred_path, "Synthetic-only Model", f"{output_dir}/confusion_matrix_synthetic.png")
plot_confusion_matrix(combined_pred_path, "Combined Model", f"{output_dir}/confusion_matrix_combined.png")

# --------------------------------------------------
# 💾 9. Save summary
# --------------------------------------------------
summary.to_csv(f"{output_dir}/summary_metrics.csv", index=False)
print(f"\n✅ All results, figures, and confusion matrices saved in: {output_dir}\n")