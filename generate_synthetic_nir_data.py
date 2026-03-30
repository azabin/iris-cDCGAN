import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from cDCGAN_model_v2 import Generator

# =====================================================
# 🧠 CONFIGURATION
# =====================================================
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# Paths
generator_path = "/Users/ananyazabin/Research/iris/iris_gan_project/results/cDCGAN_nir_v1/generator_final_nir_v1.pth"
output_root = "/Users/ananyazabin/Research/iris/iris_gan_project/results/synthetic_nir_256_v1"

# Generation parameters
num_classes = 5
class_names = ["compound", "distortion", "healthy", "opacity", "other"]
num_per_class = 500      # Number of synthetic images per class
nz = 100                 # Latent noise vector size
image_size = 256

# =====================================================
# 🧩 LOAD TRAINED GENERATOR
# =====================================================
print(f"📂 Loading generator weights from:\n{generator_path}")
G = Generator(nz=nz, num_classes=num_classes).to(device)

# ✅ Handle older checkpoint keys if saved with DataParallel
state_dict = torch.load(generator_path, map_location=device)
if "module." in list(state_dict.keys())[0]:
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        new_state_dict[k.replace("module.", "")] = v
    state_dict = new_state_dict

G.load_state_dict(state_dict)
G.eval()
print("✅ Generator loaded successfully!")

# =====================================================
# 🗂️ CREATE OUTPUT DIRECTORIES
# =====================================================
os.makedirs(output_root, exist_ok=True)
for cls in class_names:
    os.makedirs(os.path.join(output_root, cls), exist_ok=True)

# =====================================================
# 🎨 GENERATE SYNTHETIC GRAYSCALE IMAGES
# =====================================================
print(f"\n🎨 Generating {num_per_class} synthetic NIR iris images per class...\n")

with torch.no_grad():
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(output_root, class_name)
        print(f"🧩 Generating class: {class_name}")

        for i in tqdm(range(num_per_class), desc=f"{class_name:>10}"):
            # Random noise vector
            noise = torch.randn(1, nz, device=device)
            label = torch.tensor([class_idx], device=device)

            # Generate fake image (assumed single-channel output for NIR)
            fake_img = G(noise, label)

            # If the generator outputs 3 channels (trained with RGB structure), convert to grayscale
            if fake_img.shape[1] == 3:
                # Weighted average to approximate luminance: 0.299R + 0.587G + 0.114B
                fake_img = (
                    0.299 * fake_img[:, 0:1, :, :] +
                    0.587 * fake_img[:, 1:2, :, :] +
                    0.114 * fake_img[:, 2:3, :, :]
                )

            # Scale pixel values from [-1, 1] to [0, 1]
            fake_img = (fake_img + 1) / 2.0

            # Save as grayscale PNG
            filename = f"{class_name}_{i+1:03d}.png"
            save_image(fake_img, os.path.join(class_dir, filename), normalize=True)

print("\n✅ Synthetic NIR data generation complete!")
print(f"📁 Output saved in: {output_root}")
print(f"🖼️ Total images generated: {num_classes * num_per_class}")