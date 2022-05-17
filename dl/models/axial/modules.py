import torch as t
from torch import nn
from axial_attention import AxialAttention, AxialPositionalEmbedding
import ipdb
from torch.functional import F
from ...base_torch_modules.gaussian_noise import GaussianNoise


class ResidualAxialBlock(nn.Module):
    def __init__(self, embedding_dim, num_dimentions, num_heads, droupout) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads

        self.attention = AxialAttention(
            embedding_dim, num_dimentions, num_heads, 4, -1, True
        )

        self.do = nn.Dropout(droupout)
        self.act = nn.ReLU()

    def forward(self, x):
        # z = x
        x = self.attention(x)
        # x = self.res_attention(x)
        # out = self.act(x)
        out = self.do(out)
        return out


class Conv2DBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, stride, padding, droupout):
        super().__init__()
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channel)
        self.do = nn.Dropout(droupout)
        self.act = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        x = self.do(x)
        return x


class AxialClassifier(nn.Module):
    def __init__(
        self, num_classes=7, embedding_dim=32, num_heads=4, num_layers=1, dropout=0.01
    ):
        super().__init__()

        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout

        self.embedding = AxialPositionalEmbedding(self.embedding_dim, (48, 48), 1)
        self.embedding_encoder = nn.Sequential(
            nn.Linear(1, self.embedding_dim, bias=False), nn.ReLU()
        )

        self.attentions = nn.Sequential(
            AxialAttention(self.embedding_dim, 2, self.num_heads, 4, 1, True),
            nn.ReLU(),
            AxialAttention(self.embedding_dim, 2, self.num_heads, 4, 1, True),
            nn.ReLU(),
        )

        self.conv_classifier = nn.Sequential(
            Conv2DBlock(1, self.embedding_dim, 3, 1, 1, self.dropout),
            Conv2DBlock(self.embedding_dim, self.embedding_dim, 3, 1, 1, self.dropout),
            Conv2DBlock(self.embedding_dim, self.embedding_dim, 3, 1, 1, self.dropout),
        )

        self.classifier = nn.Linear(48 * 48 * 2, self.num_classes)

    def forward(self, x):
        # ipdb.set_trace()

        conv_res = self.conv_classifier(x)

        x = x.permute(0, 2, 3, 1)
        # ipdb.set_trace()
        x = self.embedding_encoder(x)
        x = x.permute(0, 3, 1, 2)
        x = self.embedding(x)
        x = self.attentions(x)

        # conv_res = conv_res.permute(0, 2, 3, 1)
        # conv_res = conv_res.view(conv_res.size(0), -1)
        # x = x.view(x.size(0), -1)
        conv_res = conv_res.max(dim=1)[0]
        x = x.max(dim=1)[0]
        x = t.cat([x, conv_res], dim=1)
        # x = x + conv_res
        x = x.view(x.size(0), -1)
        x = self.classifier(x)

        return F.softmax(x, dim=1)
