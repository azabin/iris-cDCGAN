import os
import torch
import numpy as np
import pandas as pd
import datetime
import matplotlib.pyplot as plt
from torchvision import models, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from skimage.metrics import structural_similarity as ssim
from PIL import Image
from torchvision.datasets.folder import default_loader
from scipy.linalg import sqrtm

# ----------------------------
# CONFIGURATION
# ----------------------------
real_root = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/train"
fake_root = "/Users/ananyazabin/Research/iris/iris_gan_project/results/synthetic_visible_256_v4"
save_root = "/Users/ananyazabin/Research/iris/iris_gan_project/results/metrics_visible"

os.makedirs(save_root, exist_ok=True)
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = os.path.join(save_root, f"metrics_eval_{timestamp}")
os.makedirs(output_dir, exist_ok=True)

device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# ----------------------------
# FUNCTIONS
# ----------------------------

def get_activations(path, model, batch_size=16):
    """Extract features from InceptionV3 for all images in a folder."""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    
    image_paths = [os.path.join(path, f) for f in os.listdir(path)
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    activations = []
    model.eval()
    with torch.no_grad():
        for i in tqdm(range(0, len(image_paths), batch_size), desc=f"Extracting features from {os.path.basename(path)}"):
            batch_files = image_paths[i:i+batch_size]
            imgs = [transform(default_loader(p)) for p in batch_files]
            imgs = torch.stack(imgs).to(device)
            pred = model(imgs)
            activations.append(pred.detach().cpu().numpy())
    
    activations = np.concatenate(activations, axis=0)
    return activations, image_paths

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """Calculate Frechet Inception Distance."""
    diff = mu1 - mu2
    covmean = sqrtm(sigma1.dot(sigma2))
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def random_paired_ssim(real_path, fake_path, n_pairs=50):
    """Compute random-paired SSIM between real and fake images."""
    transform = transforms.Compose([transforms.Resize((256, 256))])
    real_imgs = sorted([os.path.join(real_path, f) for f in os.listdir(real_path)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    fake_imgs = sorted([os.path.join(fake_path, f) for f in os.listdir(fake_path)
                        if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    if len(real_imgs) == 0 or len(fake_imgs) == 0:
        return np.nan
    ssim_scores = []
    for _ in range(min(n_pairs, len(real_imgs), len(fake_imgs))):
        r = Image.open(np.random.choice(real_imgs)).convert("L")
        f = Image.open(np.random.choice(fake_imgs)).convert("L")
        r, f = transform(r), transform(f)
        score = ssim(np.array(r), np.array(f), data_range=255)
        ssim_scores.append(score)
    return np.mean(ssim_scores)

# ----------------------------
# LOAD INCEPTION MODEL
# ----------------------------
inception = models.inception_v3(weights="IMAGENET1K_V1", transform_input=False)
inception.fc = torch.nn.Identity()
inception = inception.to(device)

# ----------------------------
# EVALUATION
# ----------------------------
results = []
all_real_feats, all_fake_feats = [], []

classes = sorted(os.listdir(fake_root))
print("\n🎯 Evaluating per class...\n")

for cls in classes:
    real_dir = os.path.join(real_root, cls)
    fake_dir = os.path.join(fake_root, cls)
    if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
        print(f"⚠️ Skipping class {cls}: missing data folder.")
        continue
    
    print(f"🔹 Processing class: {cls}")
    real_acts, _ = get_activations(real_dir, inception)
    fake_acts, _ = get_activations(fake_dir, inception)
    
    mu_real, sigma_real = np.mean(real_acts, axis=0), np.cov(real_acts, rowvar=False)
    mu_fake, sigma_fake = np.mean(fake_acts, axis=0), np.cov(fake_acts, rowvar=False)
    
    fid_score = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
    ssim_score = random_paired_ssim(real_dir, fake_dir)
    
    results.append({"Class": cls, "FID": fid_score, "SSIM": ssim_score})
    all_real_feats.append(real_acts)
    all_fake_feats.append(fake_acts)

# Compute overall FID
all_real_feats = np.concatenate(all_real_feats, axis=0)
all_fake_feats = np.concatenate(all_fake_feats, axis=0)
mu_real, sigma_real = np.mean(all_real_feats, axis=0), np.cov(all_real_feats, rowvar=False)
mu_fake, sigma_fake = np.mean(all_fake_feats, axis=0), np.cov(all_fake_feats, rowvar=False)
overall_fid = calculate_fid(mu_real, sigma_real, mu_fake, sigma_fake)
results.append({"Class": "Overall", "FID": overall_fid, "SSIM": np.nan})

# Save metrics
df = pd.DataFrame(results)
metrics_path = os.path.join(output_dir, "fid_ssim_results.csv")
df.to_csv(metrics_path, index=False)
print(f"\n✅ Metrics saved to: {metrics_path}")
print(df)

# ----------------------------
# VISUALIZATION (PCA + t-SNE)
# ----------------------------
print("\n🌀 Running PCA and t-SNE visualizations...")

real_feats = all_real_feats[np.random.choice(all_real_feats.shape[0], min(500, len(all_real_feats)), replace=False)]
fake_feats = all_fake_feats[np.random.choice(all_fake_feats.shape[0], min(500, len(all_fake_feats)), replace=False)]

# PCA
pca = PCA(n_components=2)
real_pca = pca.fit_transform(real_feats)
fake_pca = pca.transform(fake_feats)

plt.figure(figsize=(8, 6))
plt.scatter(real_pca[:, 0], real_pca[:, 1], alpha=0.4, label="Real", s=15)
plt.scatter(fake_pca[:, 0], fake_pca[:, 1], alpha=0.4, label="Synthetic", s=15)
plt.legend()
plt.title("PCA Projection: Real vs Synthetic Iris Features")
plt.tight_layout()
pca_path = os.path.join(output_dir, "pca_visualization.png")
plt.savefig(pca_path)
plt.close()

# t-SNE (updated for latest sklearn)
tsne = TSNE(n_components=2, perplexity=30, max_iter=2000, verbose=1)
all_feats = np.concatenate([real_feats, fake_feats])
tsne_embeds = tsne.fit_transform(all_feats)
real_embeds, fake_embeds = tsne_embeds[:len(real_feats)], tsne_embeds[len(real_feats):]

plt.figure(figsize=(8, 6))
plt.scatter(real_embeds[:, 0], real_embeds[:, 1], alpha=0.4, label="Real", s=15)
plt.scatter(fake_embeds[:, 0], fake_embeds[:, 1], alpha=0.4, label="Synthetic", s=15)
plt.legend()
plt.title("t-SNE Projection: Real vs Synthetic Iris Features")
plt.tight_layout()
tsne_path = os.path.join(output_dir, "tsne_visualization.png")
plt.savefig(tsne_path)
plt.close()

print(f"\n🖼️ PCA saved to: {pca_path}")
print(f"🖼️ t-SNE saved to: {tsne_path}")

# ----------------------------
# SUMMARY REPORT
# ----------------------------
summary_path = os.path.join(output_dir, "summary.txt")
with open(summary_path, "w") as f:
    f.write("=== Iris GAN Evaluation Summary ===\n")
    f.write(f"Generated on: {timestamp}\n\n")
    f.write("Per-Class Metrics:\n")
    f.write(df.to_string(index=False))
    f.write("\n\n")

    best_fid_class = df.loc[df['Class'] != 'Overall'].sort_values('FID').iloc[0]
    worst_fid_class = df.loc[df['Class'] != 'Overall'].sort_values('FID').iloc[-1]
    best_ssim_class = df.loc[df['Class'] != 'Overall'].sort_values('SSIM').iloc[-1]
    worst_ssim_class = df.loc[df['Class'] != 'Overall'].sort_values('SSIM').iloc[0]

    f.write(f"Best FID: {best_fid_class['Class']} ({best_fid_class['FID']:.2f})\n")
    f.write(f"Worst FID: {worst_fid_class['Class']} ({worst_fid_class['FID']:.2f})\n")
    f.write(f"Best SSIM: {best_ssim_class['Class']} ({best_ssim_class['SSIM']:.3f})\n")
    f.write(f"Worst SSIM: {worst_ssim_class['Class']} ({worst_ssim_class['SSIM']:.3f})\n")
    f.write(f"\nOverall FID: {overall_fid:.3f}\n")
    f.write(f"Total synthetic samples analyzed: {len(all_fake_feats)}\n")

print(f"📝 Summary saved to: {summary_path}")
print("🎉 Evaluation complete!")