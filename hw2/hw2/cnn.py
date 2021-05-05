import torch
import torch.nn as nn
from torch.nn import functional as f
import itertools as it

ACTIVATIONS = {"relu": nn.ReLU, "lrelu": nn.LeakyReLU}
POOLINGS = {"avg": nn.AvgPool2d, "max": nn.MaxPool2d}


class ConvClassifier(nn.Module):
    """
    A convolutional classifier model based on PyTorch nn.Modules.

    The architecture is:
    [(CONV -> ACT)*P -> POOL]*(N/P) -> (FC -> ACT)*M -> FC
    """

    def __init__(
            self,
            in_size,
            out_classes: int,
            channels: list,
            pool_every: int,
            hidden_dims: list,
            conv_params: dict = {},
            activation_type: str = "relu",
            activation_params: dict = {},
            pooling_type: str = "max",
            pooling_params: dict = {},
    ):
        """
        :param in_size: Size of input images, e.g. (C,H,W).
        :param out_classes: Number of classes to output in the final layer.
        :param channels: A list of of length N containing the number of
            (output) channels in each conv layer.
        :param pool_every: P, the number of conv layers before each max-pool.
        :param hidden_dims: List of of length M containing hidden dimensions of
            each Linear layer (not including the output layer).
        :param conv_params: Parameters for convolution layers.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        :param pooling_type: Type of pooling to apply; supports 'max' for max-pooling or
            'avg' for average pooling.
        :param pooling_params: Parameters passed to pooling layer.
        """
        super().__init__()
        assert channels and hidden_dims

        self.in_size = in_size
        self.out_classes = out_classes
        self.channels = channels
        self.pool_every = pool_every
        self.hidden_dims = hidden_dims
        self.conv_params = conv_params
        self.activation_type = activation_type
        self.activation_params = activation_params
        self.pooling_type = pooling_type
        self.pooling_params = pooling_params

        if activation_type not in ACTIVATIONS or pooling_type not in POOLINGS:
            raise ValueError("Unsupported activation or pooling type")

        self.feature_extractor = self._make_feature_extractor()
        self.classifier = self._make_classifier()

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [(CONV -> ACT)*P -> POOL]*(N/P)
        #  Apply activation function after each conv, using the activation type and
        #  parameters.
        #  Apply pooling to reduce dimensions after every P convolutions, using the
        #  pooling type and pooling parameters.
        #  Note: If N is not divisible by P, then N mod P additional
        #  CONV->ACTs should exist at the end, without a POOL after them.
        # ====== YOUR CODE: ======
        pool = POOLINGS[self.pooling_type]
        activation_f = ACTIVATIONS[self.activation_type](**self.activation_params)
        conv = nn.Conv2d

        layers += [conv(in_channels, self.channels[0], **self.conv_params)]
        layers += [activation_f]
        for p_counter, (in_, out_) in enumerate(zip(self.channels, self.channels[1:]), 2):
            layers += [conv(in_, out_, **self.conv_params)]
            layers += [activation_f]
            if p_counter % self.pool_every == 0:
                layers += [pool(**self.pooling_params)]
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def _make_classifier(self):
        layers = []
        # TODO: Create the classifier part of the model:
        #  (FC -> ACT)*M -> Linear
        #  You'll first need to calculate the number of features going in to
        #  the first linear layer.
        #  The last Linear layer should have an output dim of out_classes.
        # ====== YOUR CODE: ======
        activation_f = ACTIVATIONS[self.activation_type](**self.activation_params)
        dummy_input = torch.ones((1,) + tuple(self.in_size))
        first_in = self.feature_extractor(dummy_input).numel()

        layers += [nn.Linear(first_in, self.hidden_dims[0]), activation_f]
        for in_, out_ in zip(self.hidden_dims, self.hidden_dims[1:]):
            layers += [nn.Linear(in_, out_), activation_f]
        layers += [nn.Linear(self.hidden_dims[-1], self.out_classes)]
        # ========================
        seq = nn.Sequential(*layers)
        return seq

    def forward(self, x):
        # TODO: Implement the forward pass.
        #  Extract features from the input, run the classifier on them and
        #  return class scores.
        # ====== YOUR CODE: ======
        out = self.classifier(self.feature_extractor(x).flatten(start_dim=1))
        # ========================
        return out


class ResidualBlock(nn.Module):
    """
    A general purpose residual block.
    """

    def __init__(
            self,
            in_channels: int,
            channels: list,
            kernel_sizes: list,
            batchnorm=False,
            dropout=0.0,
            activation_type: str = "relu",
            activation_params: dict = {},
            **kwargs,
    ):
        """
        :param in_channels: Number of input channels to the first convolution.
        :param channels: List of number of output channels for each
            convolution in the block. The length determines the number of
            convolutions.
        :param kernel_sizes: List of kernel sizes (spatial). Length should
            be the same as channels. Values should be odd numbers.
        :param batchnorm: True/False whether to apply BatchNorm between
            convolutions.
        :param dropout: Amount (p) of Dropout to apply between convolutions.
            Zero means don't apply dropout.
        :param activation_type: Type of activation function; supports either 'relu' or
            'lrelu' for leaky relu.
        :param activation_params: Parameters passed to activation function.
        """
        super().__init__()
        assert channels and kernel_sizes
        assert len(channels) == len(kernel_sizes)
        assert all(map(lambda x: x % 2 == 1, kernel_sizes))

        if activation_type not in ACTIVATIONS:
            raise ValueError("Unsupported activation type")

        self.main_path, self.shortcut_path = None, None

        # TODO: Implement a generic residual block.
        #  Use the given arguments to create two nn.Sequentials:
        #  - main_path, which should contain the convolution, dropout,
        #    batchnorm, relu sequences (in this order).
        #    Should end with a final conv as in the diagram.
        #  - shortcut_path which should represent the skip-connection and
        #    may contain a 1x1 conv.
        #  Notes:
        #  - Use convolutions which preserve the spatial extent of the input.
        #  - Use bias in the main_path conv layers, and no bias in the skips.
        #  - For simplicity of implementation, assume kernel sizes are odd.
        #  - Don't create layers which you don't use! This will prevent
        #    correct comparison in the test.
        # ====== YOUR CODE: ======
        activation_f = ACTIVATIONS[activation_type](**activation_params)
        conv = nn.Conv2d
        drop = nn.Dropout2d
        norm = nn.BatchNorm2d

        layers = []

        layers += [conv(in_channels, channels[0], kernel_size=kernel_sizes[0], padding=kernel_sizes[0]//2)]
        for in_, out_, k_size in zip(channels, channels[1:], kernel_sizes[1:]):
            if dropout > 0:
                layers += [drop(dropout)]
            if batchnorm:
                layers += [norm(in_)]
            layers += [activation_f]
            layers += [conv(in_, out_, kernel_size=k_size, padding=k_size//2)]

        self.main_path = nn.Sequential(*layers)

        shortcut = []

        if in_channels != channels[-1]:
            shortcut += [conv(in_channels, channels[-1], kernel_size=1, bias=False)]
        else:
            shortcut += [nn.Identity(**kwargs)]

        self.shortcut_path = nn.Sequential(*shortcut)
        # ========================

    def forward(self, x):
        out = self.main_path(x)
        out += self.shortcut_path(x)
        out = torch.relu(out)
        return out


class ResNetClassifier(ConvClassifier):
    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=False,
            dropout=0.0,
            **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        pool = POOLINGS[self.pooling_type]
        size = self.pool_every
        channels = self.channels
        k_size = 3
        chunks = [channels[i * size:(i + 1) * size] for i in range((len(channels) + size - 1) // size )]
        layers += [ResidualBlock(in_channels, chunks[0], [k_size] * len(chunks[0]), self.batchnorm, self.dropout, self.activation_type, self.activation_params)]
        last_out = chunks[0][-1]
        for chunk in chunks[1:]:
            layers += [pool(**self.pooling_params)]
            layers += [ResidualBlock(last_out, chunk, [k_size] * len(chunk), self.batchnorm, self.dropout, self.activation_type, self.activation_params)]
            last_out = chunk[-1]
        # ========================
        seq = nn.Sequential(*layers)
        return seq


class Inception(nn.Module):
    """
    Taken from d2l.ai web inception block from GoogleNet
    """
    # `c1`--`c4` are the number of output channels for each path
    def __init__(self, in_channels, out_channels, **kwargs):
        super(Inception, self).__init__(**kwargs)

        n = out_channels // 16

        # Path 1 is a single 1 x 1 convolutional layer
        self.p1_1 = nn.Conv2d(in_channels, n * 4, kernel_size=1)
        # Path 2 is a 1 x 1 convolutional layer followed by a 3 x 3
        # convolutional layer
        self.p2_1 = nn.Conv2d(in_channels, n * 2, kernel_size=1)
        self.p2_2 = nn.Conv2d(n * 2, n * 8, kernel_size=3, padding=1)
        # Path 3 is a 1 x 1 convolutional layer followed by a 5 x 5
        # convolutional layer
        self.p3_1 = nn.Conv2d(in_channels, n, kernel_size=1)
        self.p3_2 = nn.Conv2d(n, n * 2, kernel_size=5, padding=2)
        # Path 4 is a 3 x 3 maximum pooling layer followed by a 1 x 1
        # convolutional layer
        self.p4_1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p4_2 = nn.Conv2d(in_channels, n * 2, kernel_size=1)

    def forward(self, x):
        p1 = f.relu(self.p1_1(x))
        p2 = f.relu(self.p2_2(f.relu(self.p2_1(x))))
        p3 = f.relu(self.p3_2(f.relu(self.p3_1(x))))
        p4 = f.relu(self.p4_2(self.p4_1(x)))
        # Concatenate the outputs on the channel dimension
        return torch.cat((p1, p2, p3, p4), dim=1)


class YourCodeNet(ConvClassifier):
    def __init__(
            self,
            in_size,
            out_classes,
            channels,
            pool_every,
            hidden_dims,
            batchnorm=True,
            dropout=0.4,
            **kwargs,
    ):
        """
        See arguments of ConvClassifier & ResidualBlock.
        """
        self.batchnorm = batchnorm
        self.dropout = dropout
        super().__init__(
            in_size, out_classes, channels, pool_every, hidden_dims, **kwargs
        )

    def _make_feature_extractor(self):
        in_channels, in_h, in_w, = tuple(self.in_size)

        layers = []
        # TODO: Create the feature extractor part of the model:
        #  [-> (CONV -> ACT)*P -> POOL]*(N/P)
        #   \------- SKIP ------/
        #  For the ResidualBlocks, use only dimension-preserving 3x3 convolutions.
        #  Apply Pooling to reduce dimensions after every P convolutions.
        #  Notes:
        #  - If N is not divisible by P, then N mod P additional
        #    CONV->ACT (with a skip over them) should exist at the end,
        #    without a POOL after them.
        #  - Use your own ResidualBlock implementation.
        # ====== YOUR CODE: ======
        layers = []

        layers += [nn.Conv2d(in_channels, self.channels[0], kernel_size=7, padding=3)]
        for p_counter, (in_, out_) in enumerate(zip(self.channels, self.channels[1:]), 2):
            if in_ == out_:
                layers += [nn.BatchNorm2d(in_)]
                layers += [nn.ReLU()]
                layers += [nn.Conv2d(in_, out_, kernel_size=3, padding=1)]
            else:
                layers += [Inception(in_, out_)]
            if p_counter % self.pool_every == 0:
                layers += [nn.BatchNorm2d(in_)]
                layers += [nn.ReLU()]
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        # ========================
        seq = nn.Sequential(*layers)
        return seq
