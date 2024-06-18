from __future__ import print_function

import argparse
import os
import numpy as np

from torch.autograd import Variable
from torch.autograd import grad as torch_grad
import torch

torch.autograd.set_detect_anomaly(True)

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from torchvision.utils import save_image

from itertools import chain as ichain

os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser(description="ClusterGAN Training Script")
parser.add_argument(
    "-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs"
)
parser.add_argument(
    "-b", "--batch_size", dest="batch_size", default=128, type=int, help="Batch size"
)
parser.add_argument(
    "-i",
    "--img_size",
    dest="img_size",
    type=int,
    default=28,
    help="Size of image dimension",
)
parser.add_argument(
    "-d",
    "--latent_dim",
    dest="latent_dim",
    default=256,
    type=int,
    help="Dimension of latent space",
)
parser.add_argument(
    "-l", "--lr", dest="learning_rate", type=float, default=0.0001, help="Learning rate"
)
parser.add_argument(
    "-c",
    "--n_critic",
    dest="n_critic",
    type=int,
    default=1,
    help="Number of training steps for discriminator per iter",
)
parser.add_argument(
    "-w",
    "--wass_flag",
    dest="wass_flag",
    action="store_true",
    help="Flag for Wasserstein metric",
)
args = parser.parse_args()


# Sample a random latent space vector
def sample_z(shape=64, latent_dim=10, n_c=10, fix_class=-1, req_grad=False):
    assert fix_class == -1 or (fix_class >= 0 and fix_class < n_c), (
        "Requested class %i outside bounds." % fix_class
    )

    Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    # Sample noise as generator input, zn
    zn = Variable(
        Tensor(0.75 * np.random.normal(0, 1, (shape, latent_dim))),
        requires_grad=req_grad,
    )

    # Pure one-hot vector generation
    zc_FT = Tensor(shape, n_c).fill_(0)
    zc_idx = torch.empty(shape, dtype=torch.long)

    if fix_class == -1:
        zc_idx = zc_idx.random_(n_c).cuda()
        zc_FT = zc_FT.scatter_(1, zc_idx.unsqueeze(1), 1.0)
    else:
        zc_idx[:] = fix_class
        zc_FT[:, fix_class] = 1

        zc_idx = zc_idx.cuda()
        zc_FT = zc_FT.cuda()

    zc = Variable(zc_FT, requires_grad=req_grad)

    return zn, zc, zc_idx


# calculate gradient penalty
def calc_gradient_penalty(netD, real_data, generated_data):
    LAMBDA = 10
    b_size = real_data.size()[0]

    alpha = torch.rand(b_size, 1, 1, 1, device=real_data.device)
    alpha = alpha.expand_as(real_data)

    interpolated = alpha * real_data + (1 - alpha) * generated_data
    interpolated = Variable(interpolated, requires_grad=True)

    prob_interpolated = netD(interpolated)

    gradients = torch_grad(
        outputs=prob_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones(prob_interpolated.size(), device=real_data.device),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradients = gradients.view(b_size, -1)
    gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12)

    return LAMBDA * ((gradients_norm - 1) ** 2).mean()


# Weight Initializer
def initialize_weights(net):
    for m in net.modules():
        if (
            isinstance(m, nn.Conv2d)
            or isinstance(m, nn.ConvTranspose2d)
            or isinstance(m, nn.Linear)
        ):
            nn.init.normal_(m.weight, 0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


# Softmax function
def softmax(x):
    return F.softmax(x, dim=1)


class Reshape(nn.Module):
    """
    Class for performing a reshape as a layer in a sequential model.
    """

    def __init__(self, shape=[]):
        super(Reshape, self).__init__()
        self.shape = shape

    def forward(self, x):
        return x.view(x.size(0), *self.shape)

    def extra_repr(self):
        # (Optional)Set the extra information about this module. You can test
        # it by printing an object of this class.
        return "shape={}".format(self.shape)


class Generator_CNN(nn.Module):
    """
    CNN to model the generator of a ClusterGAN
    Input is a vector from representation space of dimension z_dim
    output is a vector from image space of dimension X_dim
    """

    # Architecture : FC1024_BR-FC7x7x128_BR-(64)4dc2s_BR-(1)4dc2s_S
    def __init__(self, latent_dim, n_c, x_shape, verbose=False):
        super(Generator_CNN, self).__init__()

        self.name = "generator"
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ishape = (128, 7, 7)
        self.iels = int(np.prod(self.ishape))
        self.verbose = verbose

        self.model = nn.Sequential(
            # Fully connected layers
            nn.Linear(self.latent_dim + self.n_c, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, self.iels),
            nn.BatchNorm1d(self.iels),
            nn.LeakyReLU(0.2, inplace=True),
            # Reshape to 128 x (7x7)
            Reshape(self.ishape),
            # Upconvolution layers
            nn.ConvTranspose2d(128, 64, 4, stride=2, padding=1, bias=True),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 1, 4, stride=2, padding=1, bias=True),
            nn.Sigmoid(),
        )

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, zn, zc):
        z = torch.cat((zn, zc), 1)
        x_gen = self.model(z)
        # Reshape for output
        x_gen = x_gen.view(x_gen.size(0), *self.x_shape)
        return x_gen


class Encoder_CNN(nn.Module):
    """
    CNN to model the encoder of a ClusterGAN
    Input is vector X from image space if dimension X_dim
    Output is vector z from representation space of dimension z_dim
    """

    def __init__(self, latent_dim, n_c, verbose=False):
        super(Encoder_CNN, self).__init__()

        self.name = "encoder"
        self.channels = 1
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.cshape = (128, 5, 5)
        self.iels = int(np.prod(self.cshape))
        self.lshape = (self.iels,)
        self.verbose = verbose

        self.model = nn.Sequential(
            # Convolutional layers
            nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            # Flatten
            Reshape(self.lshape),
            # Fully connected layers
            nn.Linear(self.iels, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, latent_dim + n_c),
        )

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, in_feat):
        z_img = self.model(in_feat)
        # Reshape for output
        z = z_img.view(z_img.shape[0], -1)
        # Separate continuous and one-hot components
        zn = z[:, 0 : self.latent_dim]
        zc_logits = z[:, self.latent_dim :]
        # Softmax on the one-hot component
        zc = softmax(zc_logits)
        return zn, zc, zc_logits


class Discriminator_CNN(nn.Module):
    """
    CNN to model the discriminator of a ClusterGAN
    """

    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()

        self.name = "discriminator"
        self.channels = 1
        self.wass = wass_metric
        self.verbose = verbose

        # Architecture : 64c4s2-128c4s2_FC1024_FC1_S
        self.model = nn.Sequential(
            nn.Conv2d(self.channels, 64, 4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, stride=2, padding=1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            Reshape((128 * 7 * 7,)),
            nn.Linear(128 * 7 * 7, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 1),
        )

        initialize_weights(self)

        if self.verbose:
            print("Setting up {}...\n".format(self.name))
            print(self.model)

    def forward(self, img):
        x = self.model(img)
        # Don't apply sigmoid if using WASS loss
        if not self.wass:
            x = F.sigmoid(x)
        return x


# Initialize generator and discriminator
generator = Generator_CNN(
    latent_dim=args.latent_dim, n_c=10, x_shape=(1, args.img_size, args.img_size)
)
encoder = Encoder_CNN(latent_dim=args.latent_dim, n_c=10)
discriminator = Discriminator_CNN(wass_metric=args.wass_flag)

cuda = torch.cuda.is_available()
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

if cuda:
    generator.cuda()
    encoder.cuda()
    discriminator.cuda()

# Configure data loader
os.makedirs("../../data/mnist", exist_ok=True)
dataloader = DataLoader(
    datasets.MNIST(
        "../../data/mnist",
        train=True,
        download=True,
        transform=transforms.Compose(
            [
                transforms.Resize(args.img_size),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.1307], [0.3081]
                ),  # Normalization between -1 and 1
            ]
        ),
    ),
    batch_size=args.batch_size,
    shuffle=True,
)

# Optimizers
optimizer_G = torch.optim.Adam(
    generator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9)
)
optimizer_E = torch.optim.Adam(
    encoder.parameters(), lr=args.learning_rate, betas=(0.5, 0.9)
)
optimizer_D = torch.optim.Adam(
    discriminator.parameters(), lr=args.learning_rate, betas=(0.5, 0.9)
)

# Loss functions
bce_loss = torch.nn.BCELoss()
ce_loss = torch.nn.CrossEntropyLoss()
import matplotlib.pyplot as plt


# Loss plotting function
def plot_losses(g_loss, d_loss, e_loss, epoch):
    plt.figure(figsize=(10, 5))
    plt.title("Generator and Discriminator Loss During Training")
    plt.plot(g_loss, label="G")
    plt.plot(d_loss, label="D")
    plt.plot(e_loss, label="E")
    plt.xlabel("iterations")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig(f"images/losses_epoch.png")
    plt.close()


# Initialize lists to track losses
g_losses = []
d_losses = []
e_losses = []

# ----------
#  Training
# ----------
for epoch in range(args.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        batch_size = imgs.shape[0]

        valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)

        real_imgs = Variable(imgs.type(Tensor))

        # Train Discriminator
        optimizer_D.zero_grad()

        zn, zc, zc_idx = sample_z(shape=batch_size, latent_dim=args.latent_dim, n_c=10)
        gen_imgs = generator(zn, zc)

        real_validity = discriminator(real_imgs)
        fake_validity = discriminator(gen_imgs.detach())

        if args.wass_flag:
            gradient_penalty = calc_gradient_penalty(
                discriminator, real_imgs.data, gen_imgs.data
            )
            d_loss = (
                -torch.mean(real_validity)
                + torch.mean(fake_validity)
                + gradient_penalty
            )
        else:
            real_loss = bce_loss(real_validity, valid)
            fake_loss = bce_loss(fake_validity, fake)
            d_loss = (real_loss + fake_loss) / 2

        d_loss.backward()
        optimizer_D.step()

        optimizer_E.zero_grad()
        optimizer_G.zero_grad()

        # Train Generator and Encoder every n_critic steps
        if i % args.n_critic == 0:

            # Train Generator
            optimizer_G.zero_grad()
            gen_imgs = generator(zn, zc)
            fake_validity = discriminator(gen_imgs)

            if args.wass_flag:
                g_loss = -torch.mean(fake_validity)
            else:
                g_loss = bce_loss(fake_validity, valid)

            g_loss.backward()
            optimizer_G.step()

            # Train Encoder
            optimizer_E.zero_grad()

            zn, zc, zc_idx = sample_z(
                shape=batch_size, latent_dim=args.latent_dim, n_c=10
            )
            gen_imgs = generator(zn, zc)

            enc_zn, enc_zc, enc_zc_logits = encoder(gen_imgs)

            H = ce_loss(enc_zc_logits, zc_idx)
            EU = torch.mean(torch.sum((enc_zn - zn) ** 2, dim=1))
            e_loss = H + EU

            e_loss.backward()
            optimizer_E.step()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [E loss: %f]"
            % (
                epoch,
                args.n_epochs,
                i,
                len(dataloader),
                d_loss.item(),
                g_loss.item(),
                e_loss.item(),
            )
        )

        g_losses.append(g_loss.item())
        d_losses.append(d_loss.item())
        e_losses.append(e_loss.item())

        batches_done = epoch * len(dataloader) + i
        if batches_done % 500 == 0:
            save_image(
                gen_imgs.data[:25],
                "images/%d.png" % batches_done,
                nrow=5,
                normalize=True,
            )

    plot_losses(g_losses, d_losses, e_losses, epoch)
