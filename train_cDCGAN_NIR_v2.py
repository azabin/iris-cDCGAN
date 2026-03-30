import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
from cDCGAN_model_v3 import Generator, Discriminator

# =====================================================
# CONFIGURATION
# =====================================================
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"\n🧠 Training on: {device}")

data_root = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/NIR_processed/altogether"
output_dir = "/Users/ananyazabin/Research/iris/iris_gan_project/results/cDCGAN_nir_v2"
os.makedirs(output_dir, exist_ok=True)

batch_size = 8
image_size = 256
num_epochs = 600
nz = 100
num_classes = 5
lr = 0.0002
beta1 = 0.5

# =====================================================
# DATA LOADER
# =====================================================
transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = datasets.ImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"✅ Loaded {len(dataset)} NIR images from {data_root}")
print(f"Classes: {dataset.classes}")

# =====================================================
# MODELS
# =====================================================
G = Generator(nz=nz, num_classes=num_classes).to(device)
D = Discriminator(num_classes=num_classes).to(device)

criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

# =====================================================
# TRAINING LOOP
# =====================================================
losses_G, losses_D = [], []
epoch_log_file = os.path.join(output_dir, "epoch_losses.txt")

print("\n🚀 Starting NIR cDCGAN Training...\n")

for epoch in range(1, num_epochs + 1):
    for i, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
        imgs, labels = imgs.to(device, dtype=torch.float32), labels.to(device)
        bs = imgs.size(0)

        real_labels = torch.ones(bs, 1, device=device)
        fake_labels = torch.zeros(bs, 1, device=device)

        # Add random noise to images for stability
        imgs_noisy = imgs + 0.05 * torch.randn_like(imgs)

        # -----------------
        # Train Discriminator
        # -----------------
        D.zero_grad()
        out_real = D(imgs_noisy, labels)
        loss_real = criterion(out_real, real_labels)

        noise = torch.randn(bs, nz, device=device)
        fake_imgs = G(noise, labels)
        out_fake = D(fake_imgs.detach(), labels)
        loss_fake = criterion(out_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # -----------------
        # Train Generator
        # -----------------
        G.zero_grad()
        out_gen = D(fake_imgs, labels)
        loss_G = criterion(out_gen, real_labels)
        loss_G.backward()
        optimizerG.step()

    losses_D.append(loss_D.item())
    losses_G.append(loss_G.item())

    # Log epoch losses to file
    with open(epoch_log_file, "a") as f:
        f.write(f"Epoch {epoch}: Loss_D = {loss_D.item():.4f}, Loss_G = {loss_G.item():.4f}\n")

    print(f"Epoch [{epoch}/{num_epochs}] | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

    # Save sample images and checkpoint every 10 epochs
    if epoch % 10 == 0:
        utils.save_image(fake_imgs[:25],
                         os.path.join(output_dir, f"epoch_{epoch:03d}.png"),
                         nrow=5, normalize=True)
        torch.save({
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "optG": optimizerG.state_dict(),
            "optD": optimizerD.state_dict()
        }, os.path.join(output_dir, "checkpoint.pth"))
        print(f"💾 Checkpoint saved at epoch {epoch}")

# Save final models
torch.save(G.state_dict(), os.path.join(output_dir, "generator_final_nir_v2.pth"))
torch.save(D.state_dict(), os.path.join(output_dir, "discriminator_final_nir_v2.pth"))
print("\n✅ Training completed successfully!")
