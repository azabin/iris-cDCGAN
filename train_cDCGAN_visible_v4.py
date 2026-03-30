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
print(f"\nTraining on: {device}")

# Paths
data_root = "/Users/ananyazabin/Research/iris/iris_gan_project/data/Warsaw-BioBase-Disease-Iris-v2.1/sorted_for_gan/visible_processed/train"
pretrained_G_path = "/Users/ananyazabin/Research/iris/iris_gan_project/results/cDCGAN_visible_v2/clear/generator_final.pth"
pretrained_D_path = "/Users/ananyazabin/Research/iris/iris_gan_project/results/cDCGAN_visible_v2/clear/discriminator_final.pth"
checkpoint_path = "visible_256_results_v4/checkpoint.pth"

# Training parameters
batch_size = 8
image_size = 256
num_epochs = 600
nz = 100
num_classes = 5
lr = 0.0002
beta1 = 0.5

os.makedirs("visible_256_results_v4", exist_ok=True)

# =====================================================
# PATCHGAN-STYLE DISCRIMINATOR WRAPPER WITH SAFE SPECTRAL NORM
# =====================================================
def add_spectral_norm(module):
    """Recursively adds spectral normalization to Conv2d and Linear layers if not already applied."""
    for name, layer in module.named_children():
        if isinstance(layer, (nn.Conv2d, nn.Linear)):
            # Skip if spectral norm is already applied
            if not hasattr(layer, 'weight_u'):
                try:
                    setattr(module, name, nn.utils.spectral_norm(layer))
                except Exception as e:
                    print(f"⚠️ Skipping spectral norm on layer '{name}': {e}")
        else:
            add_spectral_norm(layer)

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

# Apply spectral normalization safely
add_spectral_norm(D)
print("✅ Spectral normalization applied safely to discriminator layers")

# =====================================================
# LOAD PRETRAINED WEIGHTS
# =====================================================
def load_pretrained(model, pretrained_path, name):
    if os.path.exists(pretrained_path):
        pretrained_dict = torch.load(pretrained_path, map_location=device)
        model_dict = model.state_dict()
        filtered_dict = {k: v for k, v in pretrained_dict.items()
                         if k in model_dict and v.size() == model_dict[k].size()}
        model_dict.update(filtered_dict)
        model.load_state_dict(model_dict)
        print(f"✅ Loaded {len(filtered_dict)} matching parameters into {name} (skipped mismatched layers).")
    else:
        print(f"⚠️ No pretrained {name} found, starting fresh.")

load_pretrained(G, pretrained_G_path, "Generator")
load_pretrained(D, pretrained_D_path, "Discriminator")

# =====================================================
# LOSS, OPTIMIZERS, AND SCHEDULERS
# =====================================================
criterion = nn.BCELoss()
optimizerD = optim.Adam(D.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = optim.Adam(G.parameters(), lr=lr, betas=(beta1, 0.999))

def lambda_rule(epoch):
    return 1.0 - max(0, epoch - 400) / float(num_epochs - 400)

schedulerD = optim.lr_scheduler.LambdaLR(optimizerD, lr_lambda=lambda_rule)
schedulerG = optim.lr_scheduler.LambdaLR(optimizerG, lr_lambda=lambda_rule)

# =====================================================
# CHECKPOINT LOAD/RESUME
# =====================================================
start_epoch = 1
losses_G, losses_D = [], []

if os.path.exists(checkpoint_path):
    print("🔄 Resuming from checkpoint...")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    G.load_state_dict(checkpoint["G"])
    D.load_state_dict(checkpoint["D"])
    optimizerG.load_state_dict(checkpoint["optG"])
    optimizerD.load_state_dict(checkpoint["optD"])
    schedulerG.load_state_dict(checkpoint["schG"])
    schedulerD.load_state_dict(checkpoint["schD"])
    losses_G = checkpoint["losses_G"]
    losses_D = checkpoint["losses_D"]
    start_epoch = checkpoint["epoch"] + 1
    print(f"✅ Resumed from epoch {start_epoch}")

# =====================================================
# SAFE OUTPUT FLATTENING FUNCTION
# =====================================================
def flatten_output(output, batch_size):
    out = output.view(batch_size, -1)
    out = out.mean(dim=1, keepdim=True)
    return out

# =====================================================
# TRAINING LOOP
# =====================================================
for epoch in range(start_epoch, num_epochs + 1):
    for i, (imgs, labels) in enumerate(tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}")):
        imgs, labels = imgs.to(device), labels.to(device)
        batch_size_curr = imgs.size(0)

        real_labels = torch.ones(batch_size_curr, 1, device=device)
        fake_labels = torch.zeros(batch_size_curr, 1, device=device)

        # Train Discriminator
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

        # Train Generator
        G.zero_grad()
        outputs = flatten_output(D(fake_imgs, labels), batch_size_curr)
        loss_G = criterion(outputs, real_labels)
        loss_G.backward()
        optimizerG.step()

    schedulerD.step()
    schedulerG.step()

    losses_D.append(loss_D.item())
    losses_G.append(loss_G.item())

    print(f"Epoch [{epoch}/{num_epochs}] | Loss_D: {loss_D.item():.4f} | Loss_G: {loss_G.item():.4f}")

    # Save samples and checkpoints every 10 epochs
    if epoch % 10 == 0:
        utils.save_image(fake_imgs[:25], f"visible_256_results_v4/epoch_{epoch:03d}.png", nrow=5, normalize=True)

        plt.figure(figsize=(8, 5))
        plt.plot(losses_D, label="Discriminator Loss")
        plt.plot(losses_G, label="Generator Loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.title("Training Loss Progress")
        plt.legend()
        plt.tight_layout()
        plt.savefig("visible_256_results_v4/loss_curve.png")
        plt.close()

        torch.save({
            "epoch": epoch,
            "G": G.state_dict(),
            "D": D.state_dict(),
            "optG": optimizerG.state_dict(),
            "optD": optimizerD.state_dict(),
            "schG": schedulerG.state_dict(),
            "schD": schedulerD.state_dict(),
            "losses_G": losses_G,
            "losses_D": losses_D
        }, checkpoint_path)
        print(f"💾 Checkpoint saved at epoch {epoch}")

# =====================================================
# SAVE FINAL MODELS
# =====================================================
torch.save(G.state_dict(), "visible_256_results_v4/generator_final_v4.pth")
torch.save(D.state_dict(), "visible_256_results_v4/discriminator_final_v4.pth")
print("✅ Training completed and models saved successfully!")