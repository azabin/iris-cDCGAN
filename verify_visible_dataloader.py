import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import torchvision.utils as vutils

# === CONFIGURATION ===
data_root = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/train"  # <-- CHANGE this path to your visible_split/train folder
image_size = 256
batch_size = 8

# === TRANSFORMS ===
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.RandomApply([transforms.ColorJitter(brightness=0.1, contrast=0.1)], p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)  # Normalize to [-1, 1]
])

# === LOAD DATA ===
dataset = datasets.ImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0) ########

print(f"✅ Loaded {len(dataset)} images from {data_root}")
print(f"📂 Classes: {dataset.classes}")

# === VISUALIZE SAMPLE BATCH ===
images, labels = next(iter(dataloader))
grid = vutils.make_grid(images, nrow=4, normalize=True)

plt.figure(figsize=(6, 6))
plt.axis("off")
plt.title("Sample Iris Images (Preprocessed)")
plt.imshow(grid.permute(1, 2, 0))
plt.show()

# === SHAPE CHECK ===
print(f"🧩 Image batch shape: {images.shape}")
print(f"🧩 Label batch shape: {labels.shape}")