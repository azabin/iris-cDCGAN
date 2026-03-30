import os
import torch
import numpy as np
from torchvision import transforms, models
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
from scipy import linalg

# -------------------------------
# Configuration
# -------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# Paths — update if needed
real_root = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/train"
synthetic_root = "/Users/ananyazabin/Research/iris/iris_gan_project/results/synthetic_visible_256_v4"
output_root = "/Users/ananyazabin/Research/iris/iris_gan_project/results/metrics_visible"

# Create dated output folder
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
result_dir = os.path.join(output_root, f"fid_ssim_results_{timestamp}")
os.makedirs(result_dir, exist_ok=True)
print(f"📁 Results will be saved in: {result_dir}")

# -------------------------------
# Utility Functions
# -------------------------------
def load_images_from_folder(folder, transform):
    """Load and transform images from a flat folder (no subfolders)."""
    images = []
    for fname in os.listdir(folder):
        if fname.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            path = os.path.join(folder, fname)
            try:
                img = Image.open(path).convert('RGB')
                images.append(transform(img))
            except Exception as e:
                print(f"⚠️ Could not load {fname}: {e}")
    return torch.stack(images) if images else torch.empty(0)

def get_activations(data_path, model, batch_size=16, dims=2048):
    """Extract features from all images in a flat folder using InceptionV3."""
    transform = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    imgs = load_images_from_folder(data_path, transform)
    if imgs.nelement() == 0:
        print(f"⚠️ No images found in {data_path}")
        return np.empty((0, dims))

    dataloader = torch.utils.data.DataLoader(imgs, batch_size=batch_size, shuffle=False)
    pred_arr = np.empty((len(imgs), dims))

    start_idx = 0
    for batch in tqdm(dataloader, desc=f"Extracting features from {os.path.basename(data_path)}"):
        batch = batch.to(device)
        with torch.no_grad():
            pred = model(batch)[0].squeeze(-1).squeeze(-1).cpu().numpy()
        pred_arr[start_idx:start_idx + pred.shape[0]] = pred
        start_idx += pred.shape[0]

    return pred_arr

def calculate_fid(mu1, sigma1, mu2, sigma2, eps=1e-6):
    """Compute Fréchet Inception Distance."""
    diff = mu1 - mu2
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    fid = diff.dot(diff) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def compute_classwise_ssim(real_dir, synth_dir):
    """Compute average SSIM per class."""
    ssim_scores = []
    for img_name in tqdm(os.listdir(real_dir), desc=f"SSIM {os.path.basename(real_dir)}"):
        real_path = os.path.join(real_dir, img_name)
        synth_path = os.path.join(synth_dir, img_name)
        if not os.path.exists(synth_path):
            continue
        real_img = np.array(Image.open(real_path).convert("L").resize((256, 256)))
        synth_img = np.array(Image.open(synth_path).convert("L").resize((256, 256)))
        ssim_val = ssim(real_img, synth_img)
        ssim_scores.append(ssim_val)
    return np.mean(ssim_scores) if ssim_scores else np.nan

# -------------------------------
# Load Pretrained InceptionV3
# -------------------------------
inception = models.inception_v3(weights=models.Inception_V3_Weights.DEFAULT, transform_input=False)
inception.fc = torch.nn.Identity()
inception.eval().to(device)

# -------------------------------
# Compute FID and SSIM per class
# -------------------------------
classes = sorted(os.listdir(real_root))
results = []

print("\n🎯 Computing FID and SSIM per class...\n")

for cls in classes:
    real_dir = os.path.join(real_root, cls)
    synth_dir = os.path.join(synthetic_root, cls)
    if not os.path.exists(synth_dir):
        print(f"⚠️ Missing synthetic class folder: {cls}")
        continue

    print(f"\n🔹 Processing class: {cls}")
    real_acts = get_activations(real_dir, inception)
    synth_acts = get_activations(synth_dir, inception)

    if real_acts.size == 0 or synth_acts.size == 0:
        print(f"⚠️ Skipping class {cls} (no valid activations).")
        continue

    mu_real, sigma_real = np.mean(real_acts, axis=0), np.cov(real_acts, rowvar=False)
    mu_synth, sigma_synth = np.mean(synth_acts, axis=0), np.cov(synth_acts, rowvar=False)
    fid_score = calculate_fid(mu_real, sigma_real, mu_synth, sigma_synth)
    ssim_score = compute_classwise_ssim(real_dir, synth_dir)

    results.append({"Class": cls, "FID": fid_score, "SSIM": ssim_score})

# -------------------------------
# Compute Overall FID
# -------------------------------
print("\n📊 Computing overall FID...\n")
real_all = np.concatenate([get_activations(os.path.join(real_root, c), inception) for c in classes if os.path.exists(os.path.join(real_root, c))])
synth_all = np.concatenate([get_activations(os.path.join(synthetic_root, c), inception) for c in classes if os.path.exists(os.path.join(synthetic_root, c))])

mu_r, sigma_r = np.mean(real_all, axis=0), np.cov(real_all, rowvar=False)
mu_s, sigma_s = np.mean(synth_all, axis=0), np.cov(synth_all, rowvar=False)
overall_fid = calculate_fid(mu_r, sigma_r, mu_s, sigma_s)
overall_ssim = np.nanmean([r["SSIM"] for r in results])

results.append({"Class": "Overall", "FID": overall_fid, "SSIM": overall_ssim})

# -------------------------------
# Save Results
# -------------------------------
df = pd.DataFrame(results)
csv_path = os.path.join(result_dir, f"fid_ssim_results_{timestamp}.csv")
txt_path = os.path.join(result_dir, f"fid_ssim_summary_{timestamp}.txt")
plot_path = os.path.join(result_dir, f"fid_ssim_plot_{timestamp}.png")

df.to_csv(csv_path, index=False)
df.to_string(open(txt_path, "w"))

# Plot FID + SSIM
fig, ax1 = plt.subplots(figsize=(8, 5))
ax2 = ax1.twinx()
ax1.bar(df["Class"], df["FID"], color="skyblue", label="FID")
ax2.plot(df["Class"], df["SSIM"], color="orange", marker="o", label="SSIM")

ax1.set_xlabel("Class")
ax1.set_ylabel("FID (lower is better)")
ax2.set_ylabel("SSIM (higher is better)")
ax1.set_title("FID & SSIM per Class (Visible Spectrum Iris GAN)")
fig.tight_layout()
plt.savefig(plot_path)
plt.close()

print("\n✅ Evaluation complete!")
print(f"📊 Results saved to: {csv_path}")
print(f"📝 Summary saved to: {txt_path}")
print(f"📈 Plot saved to: {plot_path}")