import torch
import torch.nn as nn

# =====================================================
# GENERATOR
# =====================================================
class Generator(nn.Module):
    def __init__(self, nz=100, num_classes=5, ngf=64):
        super(Generator, self).__init__()
        self.nz = nz
        self.num_classes = num_classes
        self.ngf = ngf
        self.label_emb = nn.Embedding(num_classes, num_classes)

        self.main = nn.Sequential(
            nn.ConvTranspose2d(nz + num_classes, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),

            nn.ConvTranspose2d(ngf, 1, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        # noise: [batch, nz]
        # labels: [batch]
        label_embed = self.label_emb(labels)  # [batch, num_classes]
        x = torch.cat([noise, label_embed], dim=1)  # [batch, nz + num_classes]
        x = x.unsqueeze(2).unsqueeze(3)  # reshape to [batch, nz+num_classes, 1, 1]
        out = self.main(x)
        return out


# =====================================================
# DISCRIMINATOR
# =====================================================
class Discriminator(nn.Module):
    def __init__(self, num_classes=5, ndf=64):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        in_channels = 1 + num_classes  # 1 for grayscale image + label channels

        self.main = nn.Sequential(
            nn.Conv2d(in_channels, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False)
        )

    def forward(self, img, labels):
        # img: [batch, 1, H, W]
        # labels: [batch]
        batch_size, _, H, W = img.size()
        # Convert labels to one-hot maps and expand to image size
        label_map = torch.zeros(batch_size, self.num_classes, H, W, device=img.device)
        label_map.scatter_(1, labels.view(batch_size, 1, 1, 1).expand(-1, 1, H, W), 1)
        x = torch.cat([img, label_map], dim=1)
        validity = self.main(x)
        validity = validity.view(batch_size, -1)
        validity = torch.sigmoid(validity.mean(dim=1, keepdim=True))
        return validity
