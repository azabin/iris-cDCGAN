import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils
from tqdm import tqdm
import matplotlib.pyplot as plt
from cDCGAN_model_v2 import Generator, Discriminator

# =====================================================
# CONFIGURATION
# =====================================================
device = torch.device("mps" if torch.backends.mps.is_available() else
                      "cuda" if torch.cuda.is_available() else "cpu")
print(f"Training on: {device}")

# Paths
data_root = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/train"
pretrained_G_path = "/Users/ananyazabin/Research/iris/iris_gan_project/results/cDCGAN_visible_v2/clear/generator_final.pth"
pretrained_D_path = "/Users/ananyazabin/Research/iris/iris_gan_project/results/cDCGAN_visible_v2/clear/discriminator_final.pth"

# Training parameters
batch_size = 8
image_size = 256
num_epochs = 500
nz = 100
num_classes = 5
lr = 0.0002
beta1 = 0.5

# =====================================================
# DATA LOADER
# =====================================================
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

dataset = datasets.ImageFolder(root=data_root, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
print(f"✅ Loaded {len(dataset)} images from {data_root}")
print(f"Classes: {dataset.classes}")

# =====================================================
# MODEL INITIALIZATION
# =====================================================
G = Generator(nz=nz, num_classes=num_classes).to(device)
D = Discriminator(num_classes=num_classes).to(device)

# =====================================================
# LOAD PRETRAINED WEIGHTS
# =====================================================
if os.path.exists(pretrained_G_path):
    print(f"✅ Loaded pretrained Generator from {pretrained_G_path}")
    pretrained_dict = torch.load(pretrained_G_path, map_location=device)
    model_dict = G.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(filtered_dict)
    G.load_state_dict(model_dict)
    print(f"✅ Loaded {len(filtered_dict)} matching parameters into Generator (skipped mismatched layers).")

if os.path.exists(pretrained_D_path):
    print(f"✅ Loaded pretrained Discriminator from {pretrained_D_path}")
    pretrained_dict = torch.load(pretrained_D_path, map_location=device)
    model_dict = D.state_dict()
    filtered_dict = {k: v for k, v in pretrained_dict.items()
                     if k in model_dict and v.size() == model_dict[k].size()}
    model_dict.update(filtered_dict)
    D.load_state_dict(model_dict)
    print(f"✅ Loaded {len(filtered_dict)} matching parameters into Discriminator (skipped mismatched layers).")

# =====================================================
# LOSS, OPTIMIZERS, AND LR SCHEDULERS
# =====================================================
criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

# Linear LR decay after epoch 300
def lambda_rule(epoch):
    lr_l = 1.0 - max(0, epoch - 300) / float(500 - 300)
    return lr_l

schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda_rule)
schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda_rule)

# =====================================================
# SAFE OUTPUT FLATTENING FUNCTION
# =====================================================
def flatten_output(output, batch_size):
    """
    Reshape discriminator output to [batch_size, 1]
    Handles cases like [B, 1, 8, 8] or [B*25, 1]
    """
    out = output.view(batch_size, -1)  # merge extra dims
    out = out.mean(dim=1, keepdim=True)  # average if more than 1 value per sample
    return out

# =====================================================
# TRAINING LOOP
# =====================================================
os.makedirs("visible_256_results", exist_ok=True)
losses_G, losses_D = [], []

for epoch in range(1, num_epochs + 1):
    for i, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size_curr = imgs.size(0)

        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # ---------------------
        # Train Discriminator
        # ---------------------
        D.zero_grad()

        outputs_real = flatten_output(D(imgs, labels), batch_size_curr)
        loss_real = criterion(outputs_real, real_labels)

        noise = torch.randn(batch_size_curr, nz, device=device)
        fake_imgs = G(noise, labels)
        outputs_fake = flatten_output(D(fake_imgs.detach(), labels), batch_size_curr)
        loss_fake = criterion(outputs_fake, fake_labels)

        loss_D = loss_real + loss_fake
        loss_D.backward()
        optimizerD.step()

        # ---------------------
        # Train Generator
        # ---------------------
        G.zero_grad()
        outputs = flatten_output(D(fake_imgs, labels), batch_size_curr)
        loss_G = criterion(outputs, real_labels)
        loss_G.backward()
        optimizerG.step()

    # Step the schedulers after each epoch
    schedulerD.step()
    schedulerG.step()

    # Track and display losses
    losses_D.append(loss_D.item())
    losses_G.append(loss_G.item())

    print(f"Epoch [{epoch}/{num_epochs}] | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

    # Optional: print LR progress
    if epoch % 50 == 0:
        print(f"Current LR (G): {schedulerG.get_last_lr()[0]:.6f}, (D): {schedulerD.get_last_lr()[0]:.6f}")

    # Save generated samples every 10 epochs
    if epoch % 10 == 0:
        utils.save_image(fake_imgs[:25], f"visible_256_results/epoch_{epoch:03d}.png", nrow=5, normalize=True)

        # Plot loss curves
        plt.figure(figsize=(8, 5))
        plt.plot(losses_D, label="Discriminator Loss")
        plt.plot(losses_G, label="Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Progress")
        plt.legend()
        plt.tight_layout()
        plt.savefig("visible_256_results/loss_curve.png")
        plt.close()

# =====================================================
# SAVE FINAL MODELS
# =====================================================
torch.save(G.state_dict(), "visible_256_results/generator_final_256.pth")
torch.save(D.state_dict(), "visible_256_results/discriminator_final_256.pth")
print("✅ Training completed and models saved successfully!")