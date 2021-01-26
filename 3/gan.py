import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, channels, latent_size, num_classes, hidden_dim):
        super(Generator, self).__init__()
        self.latent_size = latent_size
        self.num_classes = num_classes
        combined_size = latent_size + num_classes
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(combined_size, hidden_dim * 8, kernel_size=4, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim * 8),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 8, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 4, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(hidden_dim, channels, kernel_size=4, stride=2, padding=1, bias=False),
            nn.Tanh()
        )

    def forward(self, latent, attr):
        cat = torch.cat((latent, attr), dim=1)
        x = self.generator(cat.view(-1, self.latent_size + self.num_classes, 1, 1))
        return x


class Discriminator(nn.Module):
    def __init__(self, channels, hidden_dim):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Conv2d(channels, hidden_dim, kernel_size=4, stride=2, padding=1, bias=False),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 2, 0.8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_dim * 2, hidden_dim * 4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 4, 0.8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_dim * 4, hidden_dim * 8, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(hidden_dim * 8, 0.8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout2d(0.25),
            nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, image):
        result = self.discriminator(image)
        return result.view(-1)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)
