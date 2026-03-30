import os
from PIL import Image
from collections import Counter

# -----------------------------
# CONFIGURATION
# -----------------------------
visible_dir = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/images-visible"
nir_dir = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/images-NIR"


def get_stats(root_dir):
    total_images = 0
    total_size = 0
    widths, heights = [], []
    eye_ids = set()
    camera_counter = Counter()

    for person_dir in os.listdir(root_dir):
        full_path = os.path.join(root_dir, person_dir)
        if not os.path.isdir(full_path):
            continue

        for fname in os.listdir(full_path):
            if not (fname.lower().endswith((".jpg", ".bmp"))):
                continue

            total_images += 1
            total_size += os.path.getsize(os.path.join(full_path, fname))

            # Try to read image dimensions
            try:
                img = Image.open(os.path.join(full_path, fname))
                widths.append(img.width)
                heights.append(img.height)
            except:
                continue

            # Extract person + eye ID
            parts = fname.split("_")
            if len(parts) >= 2:
                eye_id = parts[0] + parts[1][0].upper()  # e.g., 0060L
                eye_ids.add(eye_id)

            # Extract camera code (e.g., CA, TC, IG)
            if len(parts) >= 3:
                cam_code = parts[2].upper()
                camera_counter[cam_code] += 1

    avg_width = round(sum(widths) / len(widths)) if widths else 0
    avg_height = round(sum(heights) / len(heights)) if heights else 0
    total_size_MB = round(total_size / (1024**2), 2)

    return {
        "total_images": total_images,
        "unique_eyes": len(eye_ids),
        "avg_resolution": (avg_width, avg_height),
        "min_resolution": (min(widths), min(heights)) if widths else (0, 0),
        "max_resolution": (max(widths), max(heights)) if widths else (0, 0),
        "total_size_MB": total_size_MB,
        "camera_distribution": dict(camera_counter)
    }


# -----------------------------
# RUN ANALYSIS
# -----------------------------
print("🔍 Analyzing dataset... Please wait.\n")

visible_stats = get_stats(visible_dir)
nir_stats = get_stats(nir_dir)

print("📊 Dataset Statistics Summary:\n")

print("🟦 Visible Band:")
for k, v in visible_stats.items():
    print(f"  {k}: {v}")

print("\n⬛ NIR Band:")
for k, v in nir_stats.items():
    print(f"  {k}: {v}")

print("\n✅ Analysis complete.")