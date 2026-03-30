import os
import torch
from torchvision.utils import save_image
from tqdm import tqdm
from cDCGAN_model_v2 import Generator

# -------------------------------
# Configuration
# -------------------------------
device = torch.device("mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu")
print(f"🧠 Using device: {device}")

# Paths (update if needed)
generator_path = "/Users/ananyazabin/Research/iris/iris_gan_project/results/visible_256_results_v4/generator_final_v4.pth"
output_root = "/Users/ananyazabin/Research/iris/iris_gan_project/results/synthetic_visible_256_v4"

# Generation parameters
num_classes = 5
class_names = ["compound", "distortion", "healthy", "opacity", "other"]
num_per_class = 500   # Number of synthetic images per class
nz = 100              # Latent noise dimension
image_size = 256

# -------------------------------
# Load Trained Generator
# -------------------------------
print(f"📂 Loading generator weights from:\n{generator_path}")
G = Generator(nz=nz, num_classes=num_classes).to(device)
G.load_state_dict(torch.load(generator_path, map_location=device))
G.eval()
print("✅ Generator loaded successfully!")

# -------------------------------
# Create Output Directories
# -------------------------------
os.makedirs(output_root, exist_ok=True)
for cls in class_names:
    os.makedirs(os.path.join(output_root, cls), exist_ok=True)

# -------------------------------
# Generate Synthetic Images
# -------------------------------
print(f"\n🎨 Generating {num_per_class} synthetic iris images per class...\n")

with torch.no_grad():
    for class_idx, class_name in enumerate(class_names):
        class_dir = os.path.join(output_root, class_name)
        print(f"🧩 Generating class: {class_name}")

        for i in tqdm(range(num_per_class), desc=f"{class_name:>10}"):
            # Random noise vector
            noise = torch.randn(1, nz, device=device)
            # Class label tensor
            label = torch.tensor([class_idx], device=device)

            # Generate image
            fake_img = G(noise, label)
            fake_img = (fake_img + 1) / 2.0  # Scale from [-1,1] to [0,1]

            # Save image
            filename = f"{class_name}_{i+1:03d}.png"
            save_image(fake_img, os.path.join(class_dir, filename))

print("\n✅ Synthetic data generation complete!")
print(f"📁 Output saved in: {output_root}")
print(f"🖼️ Total images generated: {num_classes * num_per_class}")