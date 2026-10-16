"""Models and losses. Every model shares the same convolutional trunk:

    image 3x64x64 -> conv 32 -> 64 -> 128 -> 256 (each halves the size) -> 256x4x4 -> 4096 numbers

An autoencoder adds a bottleneck (a small code) and a decoder that mirrors the trunk.
The classifier adds a label head instead of a decoder.
"""

import torch
from torch import nn

FEATURES = 256 * 4 * 4  # 4096 numbers out of the trunk


def conv_trunk():
    """3x64x64 image -> 4096 features. Each block halves the size and adds channels."""
    layers, ch = [], 3
    for out in (32, 64, 128, 256):  # 64 -> 32 -> 16 -> 8 -> 4 pixels wide
        layers += [nn.Conv2d(ch, out, 4, stride=2, padding=1), nn.BatchNorm2d(out), nn.LeakyReLU(0.2)]
        ch = out
    return nn.Sequential(*layers, nn.Flatten())


class Decoder(nn.Module):
    """Code -> 3x64x64 image: the trunk run backwards with transposed convolutions."""

    def __init__(self, latent):
        super().__init__()
        self.fc = nn.Linear(latent, FEATURES)
        layers, ch = [nn.Unflatten(1, (256, 4, 4)), nn.LeakyReLU(0.2)], 256
        for out in (128, 64, 32):  # 4 -> 8 -> 16 -> 32 pixels wide
            layers += [nn.ConvTranspose2d(ch, out, 4, stride=2, padding=1), nn.BatchNorm2d(out), nn.LeakyReLU(0.2)]
            ch = out
        # Last layer: back to 64x64 RGB, squashed into [0, 1] like the input pixels.
        layers += [nn.ConvTranspose2d(ch, 3, 4, stride=2, padding=1), nn.Sigmoid()]
        self.net = nn.Sequential(*layers)

    def forward(self, z):
        return self.net(self.fc(z))


class AutoEncoder(nn.Module):
    """Plain autoencoder: squeeze 12,288 pixels through a bottleneck of `latent` numbers.

    The bottleneck is the whole point: the network cannot copy the image,
    so it has to learn what matters about cat faces to rebuild them.
    With relu_code=True the code is non-negative, which the sparse AE needs
    so that "off" means exactly zero.
    """

    def __init__(self, latent, relu_code=False):
        super().__init__()
        self.trunk = conv_trunk()
        self.to_code = nn.Linear(FEATURES, latent)
        self.relu_code = relu_code
        self.decoder = Decoder(latent)

    def encode(self, x):
        z = self.to_code(self.trunk(x))
        return torch.relu(z) if self.relu_code else z

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z), z


class VAE(nn.Module):
    """Variational autoencoder: the encoder outputs a cloud (mu, sigma), not a point.

    We sample the code from that cloud, and the KL term pulls every cloud
    towards N(0, 1). The latent space then has no holes, so decoding any
    z ~ N(0, 1) gives a cat.
    """

    def __init__(self, latent):
        super().__init__()
        self.trunk = conv_trunk()
        self.to_mu = nn.Linear(FEATURES, latent)      # centre of the cloud
        self.to_logvar = nn.Linear(FEATURES, latent)  # log sigma^2: any real number, so no sign issues
        self.decoder = Decoder(latent)

    def encode(self, x):
        h = self.trunk(x)
        return self.to_mu(h), self.to_logvar(h)

    @staticmethod
    def reparameterize(mu, logvar):
        """z = mu + sigma * eps with eps ~ N(0, 1).

        The randomness lives in eps, an outside input, so gradients can flow
        through mu and sigma. Sampling z directly would block them.
        """
        eps = torch.randn_like(mu)
        return mu + torch.exp(0.5 * logvar) * eps

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


class CNNClassifier(nn.Module):
    """The autoencoder's twin: the same encoder trunk, plus a label head instead of a decoder.

    encoder + decoder -> image (autoencoder)
    encoder + label head -> class (this model)
    """

    def __init__(self, num_classes=3):
        super().__init__()
        self.trunk = conv_trunk()
        self.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(FEATURES, num_classes))

    def forward(self, x):
        return self.head(self.trunk(x))  # logits; softmax turns them into probabilities


def reconstruction_loss(x_hat, x):
    """Squared error summed over all 12,288 pixel values of an image, averaged over the batch."""
    return ((x_hat - x) ** 2).flatten(1).sum(1).mean()


def kl_divergence(mu, logvar):
    """KL( N(mu, sigma^2) || N(0, 1) ) in closed form, summed over latent dims, averaged over the batch.

    It is zero only when mu = 0 and sigma = 1: it pays a price for every cloud
    that drifts away from the centre or shrinks to a point.
    """
    return (-0.5 * (1 + logvar - mu ** 2 - logvar.exp())).sum(1).mean()
