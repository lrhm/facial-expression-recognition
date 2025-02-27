from torch import batch_norm, nn
import torch as t
from torch.functional import F
from ...base_torch_modules.gaussian_noise import GaussianNoise

# ConvBlock with BatchNorm, Dropout and Activation
class ConvBlock(nn.Module):
    def __init__(
        self,
        c_in,
        c_out,
        kernel_size=3,
        stride=1,
        padding="same",
        bias=True,
        droupout=0.1,
        act=nn.ReLU(inplace=True),
        bn=True,
    ):
        super().__init__()

        self.conv = nn.Conv2d(
            c_in,
            c_out,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(c_out) if bn else None
        self.act = act
        self.do = nn.Dropout(droupout) if droupout > 0.0 else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.act is not None:
            x = self.act(x)
        if self.do is not None:
            x = self.do(x)
        return x


# A conv2d classifier
class Conv2dClassifier(nn.Module):
    def __init__(self, c_in=1, num_classes=7):
        super().__init__()

        self.noise_layer = GaussianNoise(0.05)
        self.hidden_dim = 64

        # 48x48x1
        self.conv1 = ConvBlock(
            c_in, self.hidden_dim, kernel_size=3, stride=1, padding=1, droupout=0.0
        )
        # 48x48x64
        self.conv2 = ConvBlock(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            droupout=0.01,
        )
        # 24x24x64
        self.conv3 = ConvBlock(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            droupout=0.01,
        )
        # 12x12x64
        self.conv4 = ConvBlock(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=3,
            stride=2,
            padding=1,
            droupout=0.01,
        )
        # 6x6x64

        self.classifier = nn.Sequential(
            nn.Linear(self.hidden_dim * 6 * 6, num_classes), nn.ReLU()
        )

    def forward(self, x):

        if self.training:
            x = self.noise_layer(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        # x = F.softmax(x, dim=1)
        return x
