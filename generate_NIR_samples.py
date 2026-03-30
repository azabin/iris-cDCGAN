import os
import torch
from torchvision.utils import save_image
from cDCGAN_model_v3 import Generator

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
generator_path = "/Users/ananyazabin/Research/iris/iris_gan_project/results/cDCGAN_nir_v2/generator_final_nir_v2.pth"
output_root = "/Users/ananyazabin/Research/iris/iris_gan_project/results/synthetic_nir_v2"
os.makedirs(output_root, exist_ok=True)

num_classes = 5
class_names = ["compound", "distortion", "healthy", "opacity", "other"]
nz = 100
num_per_class = 200

G = Generator(nz=nz, num_classes=num_classes).to(device)
G.load_state_dict(torch.load(generator_path, map_location=device))
G.eval()

with torch.no_grad():
    for class_idx, class_name in enumerate(class_names):
        out_dir = os.path.join(output_root, class_name)
        os.makedirs(out_dir, exist_ok=True)
        for i in range(num_per_class):
            z = torch.randn(1, nz, device=device)
            label = torch.tensor([class_idx], device=device)
            fake = G(z, label)
            fake = (fake + 1) / 2.0
            save_image(fake, os.path.join(out_dir, f"{class_name}_{i+1:03d}.png"))

print(f"✅ {num_per_class*num_classes} synthetic images saved in {output_root}")