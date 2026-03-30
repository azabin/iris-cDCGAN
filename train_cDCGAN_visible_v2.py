import sys, os
sys.path.append(os.path.dirname(__file__))

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from cDCGAN_model import Generator, Discriminator

# ==========================
# CONFIGURATION
# ==========================
device = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Training on: {device}")

data_root = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/clean-train"
save_dir = "/Users/ananyazabin/Research/iris/iris_gan_project/results/cDCGAN_visible_v2/clear/"
os.makedirs(save_dir, exist_ok=True)

batch_size = 16
image_size = 128   # 👈 you can later change this to 256 for final runs
num_epochs = 200
nz = 100           # latent vector size
num_classes = 5
lr = 0.0002
beta1 = 0.5

# ==========================
# DATASET & TRANSFORMS
# ==========================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(root=data_root, transform=transform)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)

print(f"✅ Loaded {len(dataset)} images from {data_root}")
print(f"Classes: {dataset.classes}")

# ==========================
# MODEL INITIALIZATION
# ==========================
netG = Generator(nz=nz, num_classes=num_classes, img_size=image_size).to(device)
netD = Discriminator(num_classes=num_classes, img_size=image_size).to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=lr, betas=(beta1, 0.999))

fixed_noise = torch.randn(25, nz, device=device)
fixed_labels = torch.tensor([i % num_classes for i in range(25)], device=device)

# ==========================
# TRAINING LOOP
# ==========================
for epoch in range(num_epochs):
    for i, (real_imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")):
        real_imgs, labels = real_imgs.to(device), labels.to(device)
        b_size = real_imgs.size(0)

        # === Train Discriminator ===
        netD.zero_grad()

        # --- Real images ---
        output_real = netD(real_imgs, labels)
        real_label = torch.ones_like(output_real, device=device)
        loss_real = criterion(output_real, real_label)

        # --- Fake images ---
        noise = torch.randn(b_size, nz, device=device)
        fake_imgs = netG(noise, labels)
        output_fake = netD(fake_imgs.detach(), labels)
        fake_label = torch.zeros_like(output_fake, device=device)
        loss_fake = criterion(output_fake, fake_label)

        # --- Combine & update D ---
        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # === Train Generator ===
        netG.zero_grad()
        output = netD(fake_imgs, labels)
        real_label = torch.ones_like(output, device=device)
        loss_G = criterion(output, real_label)
        loss_G.backward()
        optimizerG.step()

    # === Save samples every 10 epochs ===
    if (epoch + 1) % 10 == 0:
        with torch.no_grad():
            fake = netG(fixed_noise, fixed_labels).detach().cpu()
        utils.save_image(fake, os.path.join(save_dir, f"epoch_{epoch+1:03d}.png"),
                         normalize=True, nrow=5)
        print(f"🖼️ Saved generated samples at epoch {epoch+1}")

# ==========================
# SAVE FINAL MODELS
# ==========================
print("✅ Training complete!")
torch.save(netG.state_dict(), os.path.join(save_dir, "generator_final.pth"))
torch.save(netD.state_dict(), os.path.join(save_dir, "discriminator_final.pth"))
print(f"✅ Models saved to: {save_dir}")

# /Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/clean-train