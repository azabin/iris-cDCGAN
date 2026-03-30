import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm

# --------------------------------------------
# Generator for 256x256 conditional DCGAN (v2)
# --------------------------------------------
class Generator(nn.Module):
    def __init__(self, nz=100, nc=3, num_classes=5, ngf=64, embed_dim=50, use_spectral_norm=True):
        super(Generator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)
        input_dim = nz + embed_dim

        def block(in_feat, out_feat, kernel_size, stride, padding, final=False):
            layers = [
                nn.ConvTranspose2d(in_feat, out_feat, kernel_size, stride, padding, bias=False),
                nn.BatchNorm2d(out_feat),
                nn.LeakyReLU(0.1, inplace=True) if not final else nn.Tanh()
            ]
            return layers

        self.main = nn.Sequential(
            *block(input_dim, ngf * 16, 4, 1, 0),      # 1x1 -> 4x4
            *block(ngf * 16, ngf * 8, 4, 2, 1),        # 4x4 -> 8x8
            *block(ngf * 8, ngf * 4, 4, 2, 1),         # 8x8 -> 16x16
            *block(ngf * 4, ngf * 2, 4, 2, 1),         # 16x16 -> 32x32
            *block(ngf * 2, ngf, 4, 2, 1),             # 32x32 -> 64x64
            *block(ngf, ngf // 2, 4, 2, 1),            # 64x64 -> 128x128
            *block(ngf // 2, nc, 4, 2, 1, final=True)  # 128x128 -> 256x256
        )

    def forward(self, noise, labels):
        label_embedding = self.label_emb(labels)
        gen_input = torch.cat((noise, label_embedding), -1)
        out = gen_input.unsqueeze(2).unsqueeze(3)
        img = self.main(out)
        return img


# --------------------------------------------
# Discriminator for 256x256 conditional DCGAN (v2)
# --------------------------------------------
class Discriminator(nn.Module):
    def __init__(self, nc=3, num_classes=5, ndf=64, embed_dim=50, use_spectral_norm=True):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, embed_dim)

        def sn(layer):
            return spectral_norm(layer) if use_spectral_norm else layer

        def block(in_feat, out_feat, kernel_size, stride, padding, bn=True):
            layers = [sn(nn.Conv2d(in_feat, out_feat, kernel_size, stride, padding, bias=False))]
            if bn:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.main = nn.Sequential(
            *block(nc + 1, ndf, 4, 2, 1, bn=False),   # 256 -> 128
            *block(ndf, ndf * 2, 4, 2, 1),            # 128 -> 64
            *block(ndf * 2, ndf * 4, 4, 2, 1),        # 64 -> 32
            *block(ndf * 4, ndf * 8, 4, 2, 1),        # 32 -> 16
            *block(ndf * 8, ndf * 16, 4, 2, 1),       # 16 -> 8
            sn(nn.Conv2d(ndf * 16, 1, 4, 1, 0, bias=False)),  # 8 -> 1
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        # Embed labels and expand to match image spatial size
        label_embedding = self.label_emb(labels)
        label_map = label_embedding.unsqueeze(2).unsqueeze(3).expand(-1, -1, img.size(2), img.size(3))
        label_map = torch.mean(label_map, dim=1, keepdim=True)  # single channel map
        x = torch.cat((img, label_map), 1)
        validity = self.main(x)
        return validity.view(-1, 1)