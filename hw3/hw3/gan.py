import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable
from torch.utils.data import DataLoader
from torch.optim.optimizer import Optimizer
from hw3.autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        #  To extract image features you can use the EncoderCNN from the VAE
        #  section or implement something new.
        #  You can then use either an affine layer or another conv layer to
        #  flatten the features.
        # ====== YOUR CODE: ======
        layers = []
        in_channel, H, W = in_size
        in_size = 1024 * (H // 16) * (W // 16)
        self.flatten = nn.Linear(in_size, 1)

        k = [128, 256, 512, 1024]
        filters = [in_channel] + k

        conv = nn.Conv2d
        activation = nn.ReLU
        norm = nn.BatchNorm2d

        for in_, out_ in zip(filters, filters[1:]):
            layers += [conv(in_, out_, kernel_size=5, stride=2, padding=2)]
            layers += [norm(out_)]
            layers += [activation()]

        self.encoder = nn.Sequential(*layers)
        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        #  No need to apply sigmoid to obtain probability - we'll combine it
        #  with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        h = self.encoder(x)
        h = h.view(x.size(0), -1)
        y = self.flatten(h)
        # ========================
        return y


class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim

        # TODO: Create the generator model layers.
        #  To combine image features you can use the DecoderCNN from the VAE
        #  section or implement something new.
        #  You can assume a fixed image size.
        # ====== YOUR CODE: ======

        conv = nn.ConvTranspose2d
        activation = nn.LeakyReLU
        norm = nn.BatchNorm2d

        k = [1024, 512, 256, 128]
        self.img_size = (k[0], featuremap_size, featuremap_size)
        layers = []
        for in_, out_ in zip(k, k[1:]):
            layers += [conv(in_, out_, stride=2, kernel_size=5, padding=2, output_padding=1)]
            layers += [norm(out_)]
            layers += [activation()]

        layers += [nn.ConvTranspose2d(k[-1], out_channels, stride=2, kernel_size=5, padding=2, output_padding=1)]
        layers += [nn.Tanh()]

        self.decoder = nn.Sequential(*layers)
        self.linear = nn.Linear(z_dim, featuremap_size ** 2 * k[0])
        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should be part of the
        generator's computation graph or standalone tensors (i.e. should be
        be able to backprop into them and compute their gradients).
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        #  Generate n latent space samples and return their reconstructions.
        #  Don't use a loop.
        # ====== YOUR CODE: ======
        with torch.set_grad_enabled(with_grad):
            z = torch.randn((n, self.z_dim), device=device)
            samples = self.forward(z)
        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        #  Don't forget to make sure the output instances have the same
        #  dynamic range as the original (real) images.
        # ====== YOUR CODE: ======
        x = self.linear(z)
        x = self.decoder(x.view(x.shape[0], *self.img_size))
        # ========================
        return x


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the discriminator loss.
    #  See pytorch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    loss_fn = nn.BCEWithLogitsLoss()
    label_noise_delta = label_noise / 2
    y_data_noise = torch.ones(y_data.shape).to(y_data.device)
    y_generated_noise = torch.ones(y_generated.shape).to(y_generated.device)
    y_data_noise.uniform_(data_label - label_noise_delta, data_label + label_noise_delta)
    y_generated_noise.uniform_(1 - data_label - label_noise_delta, 1 - data_label + label_noise_delta)
    loss_data, loss_generated = loss_fn(y_data, y_data_noise), loss_fn(y_generated, y_generated_noise)
    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    assert data_label == 1 or data_label == 0
    # TODO:
    #  Implement the Generator loss.
    #  Think about what you need to compare the input to, in order to
    #  formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    loss_fn = nn.BCEWithLogitsLoss()
    generated_data_labels = torch.ones(y_generated.shape).to(y_generated.device) * data_label
    loss = loss_fn(y_generated, generated_data_labels)
    # ========================
    return loss


def train_batch(
    dsc_model: Discriminator,
    gen_model: Generator,
    dsc_loss_fn: Callable,
    gen_loss_fn: Callable,
    dsc_optimizer: Optimizer,
    gen_optimizer: Optimizer,
    x_data: DataLoader,
):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    #  1. Show the discriminator real and generated data
    #  2. Calculate discriminator loss
    #  3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()
    generated_data = gen_model.sample(x_data.shape[0])
    real_data_prob = dsc_model(x_data)
    gen_data_prob = dsc_model(generated_data)
    dsc_loss = dsc_loss_fn(real_data_prob, gen_data_prob)
    dsc_loss.backward()
    dsc_optimizer.step()
    # ========================

    # TODO: Generator update
    #  1. Show the discriminator generated data
    #  2. Calculate generator loss
    #  3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()
    generated_data = gen_model.sample(x_data.shape[0], with_grad=True)
    gen_data_prob = dsc_model(generated_data)
    gen_loss = gen_loss_fn(gen_data_prob)
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()


def save_checkpoint(gen_model, dsc_losses, gen_losses, checkpoint_file):
    """
    Saves a checkpoint of the generator, if necessary.
    :param gen_model: The Generator model to save.
    :param dsc_losses: Avg. discriminator loss per epoch.
    :param gen_losses: Avg. generator loss per epoch.
    :param checkpoint_file: Path without extension to save generator to.
    """

    saved = False
    checkpoint_file = f"{checkpoint_file}.pt"

    # TODO:
    #  Save a checkpoint of the generator model. You can use torch.save().
    #  You should decide what logic to use for deciding when to save.
    #  If you save, set saved to True.
    # ====== YOUR CODE: ======
    weight = 0.7
    loss_score = lambda loss_a, loss_b: loss_a * weight + loss_b * (1 - weight)
    threshold = loss_score(dsc_losses[-1], gen_losses[-1])
    for dsc_loss, gen_loss in zip(dsc_losses, gen_losses):
        if loss_score(dsc_loss, gen_loss) < threshold:
            torch.save(gen_model, checkpoint_file)
            saved = True
    # ========================

    return saved
