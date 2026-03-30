import torch
import torch.nn as nn

# ======================
# Generator
# ======================
class Generator(nn.Module):
    def __init__(self, nz=100, num_classes=5, img_size=128, channels=3):
        super(Generator, self).__init__()
        self.nz = nz
        self.img_size = img_size
        self.label_emb = nn.Embedding(num_classes, num_classes)

        input_dim = nz + num_classes

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.ConvTranspose2d(in_feat, out_feat, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.ReLU(True))
            return layers

        self.model = nn.Sequential(
            # Input is Z + label embedding
            nn.ConvTranspose2d(input_dim, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),

            *block(512, 256),
            *block(256, 128),
            *block(128, 64),
            nn.ConvTranspose2d(64, channels, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, noise, labels):
        label_input = self.label_emb(labels)
        gen_input = torch.cat((noise, label_input), -1)
        out = gen_input.view(gen_input.size(0), self.nz + self.label_emb.num_embeddings, 1, 1)
        img = self.model(out)
        return img


# ======================
# Discriminator
# ======================
class Discriminator(nn.Module):
    def __init__(self, num_classes=5, img_size=128, channels=3):
        super(Discriminator, self).__init__()
        self.label_emb = nn.Embedding(num_classes, num_classes)

        def block(in_feat, out_feat, normalize=True):
            layers = [nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_feat))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        self.model = nn.Sequential(
            nn.Conv2d(channels + 1, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            *block(64, 128),
            *block(128, 256),
            *block(256, 512),
            nn.Conv2d(512, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, img, labels):
        label_input = self.label_emb(labels).unsqueeze(2).unsqueeze(3)
        label_input = label_input.expand(labels.size(0), self.label_emb.num_embeddings, img.size(2), img.size(3))
        d_in = torch.cat((img, label_input[:, :1, :, :]), 1)
        out = self.model(d_in)
        return out.view(-1, 1)