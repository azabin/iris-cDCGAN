import os
import cv2
import numpy as np
from tqdm import tqdm

# ==============================================
# CONFIGURATION
# ==============================================
input_root = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/NIR/altogether"
output_root = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/CLAHE_enhanced_uncropped_NIR"
os.makedirs(output_root, exist_ok=True)

# CLAHE parameters
clip_limit = 2.0       # Try 2.0–3.0 for stronger enhancement
tile_grid_size = (8, 8)  # Local histogram tile size

# Optional: downscale large images if they exceed 1024px width (to prevent memory issues)
MAX_DIM = 1024  # set None if you want absolutely full res

# ==============================================
# HELPER FUNCTION
# ==============================================
def enhance_image_CLAHE(img_path, clahe, max_dim=None):
    """Load a grayscale image, optionally resize, apply CLAHE, and return enhanced image."""
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Could not read image: {img_path}")

    # Optionally resize for memory safety
    if max_dim is not None:
        h, w = img.shape
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # Apply CLAHE
    enhanced = clahe.apply(img)
    return img, enhanced

# ==============================================
# PROCESS ALL IMAGES
# ==============================================
clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
contrast_stats = []

for cls in sorted(os.listdir(input_root)):
    cls_path = os.path.join(input_root, cls)
    if not os.path.isdir(cls_path):
        continue

    out_cls_path = os.path.join(output_root, cls)
    os.makedirs(out_cls_path, exist_ok=True)

    for fname in tqdm(sorted(os.listdir(cls_path)), desc=f"Processing {cls}"):
        if not fname.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):
            continue

        in_path = os.path.join(cls_path, fname)
        out_path = os.path.join(out_cls_path, os.path.splitext(fname)[0] + "_clahe.png")

        try:
            img_orig, img_clahe = enhance_image_CLAHE(in_path, clahe, MAX_DIM)
            cv2.imwrite(out_path, img_clahe)

            # Record brightness/contrast stats for comparison
            contrast_stats.append({
                "class": cls,
                "file": fname,
                "mean_before": float(np.mean(img_orig)),
                "std_before": float(np.std(img_orig)),
                "mean_after": float(np.mean(img_clahe)),
                "std_after": float(np.std(img_clahe))
            })

        except Exception as e:
            print(f"⚠️ Skipped {fname}: {e}")

# ==============================================
# SAVE STATS CSV FOR ANALYSIS
# ==============================================
import pandas as pd
df = pd.DataFrame(contrast_stats)
csv_path = os.path.join(output_root, "contrast_stats.csv")
df.to_csv(csv_path, index=False)
print(f"\n✅ CLAHE preprocessing complete! Results saved to: {output_root}")
print(f"📊 Contrast statistics saved to: {csv_path}")

# ==============================================
# SUMMARY STATS
# ==============================================
mean_gain = (df["std_after"] - df["std_before"]).mean()
print(f"✨ Average local contrast gain: {mean_gain:.2f}")
