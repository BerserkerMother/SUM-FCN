from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F


class SumFCNAttention(nn.Module):
    def __init__(self, cfg, num_class: int = 2):
        super(SumFCNAttention, self).__init__()

        # model layers
        # project frames to lower dimension
        self.feature_projection = nn.Linear(1024, 256)

        # make pool holders
        self.pool_holders = [LayerHolder() for _ in range(5)]
        # encoder
        self.vgg16_1d = make_layers(cfg=cfg, pool_holders=self.pool_holders)
        self.conv8 = nn.Sequential(
            nn.Conv1d(512, num_class, kernel_size=1),
            nn.BatchNorm1d(num_class)
        )

        # decoder
        self.de_conv1 = nn.ConvTranspose1d(num_class, num_class,
                                           kernel_size=4, padding=1,
                                           stride=2, bias=False)
        self.de_conv2 = nn.ConvTranspose1d(num_class, num_class,
                                           kernel_size=16, stride=16,
                                           bias=False)

        # pool4 transformation
        self.pool4_transformation = nn.Sequential(
            nn.Conv1d(in_channels=512, out_channels=num_class, kernel_size=1),
            nn.BatchNorm1d(num_class)
        )

    def forward(self, x):
        batch_size = x.size()[0]

        # let the magic begin
        x = self.feature_projection(x).transpose(1, 2)
        # feed
        x = self.vgg16_1d(x)
        x = self.conv8(x)
        x = self.de_conv1(x)

        # get pool4, transform it and sum it up with de conv1
        pool4 = self.pool4_transformation(
            self.pool_holders[3].contained_tensor)

        # last de conv layer
        x = self.de_conv2(x + pool4)

        # apply softmax
        logits = F.softmax(x.transpose(1, 2).view(-1, 2), dim=-1)
        return logits


# this class only hold tensors, let us access to pool output in sequential
class LayerHolder(nn.Module):
    def __init__(self):
        super(LayerHolder, self).__init__()
        self.contained_tensor = None
        self.is_pos = False

    def in_pos(self):
        self.is_pos = True
        return self

    def forward(self, x):
        self.contained_tensor = x
        return x


def make_layers(cfg, pool_holders, batch_norm: bool = True) -> nn.Sequential:
    layers = []
    in_channels = 256
    i = 0  # index of pool holder
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool1d(kernel_size=2, stride=2),
                       pool_holders[i].in_pos()]
            i += 1
        elif v == 'E':
            conv1d = nn.Conv1d(in_channels, in_channels, kernel_size=3,
                               padding=1)
            layers += [conv1d, nn.ReLU(inplace=True), nn.Dropout(p=0.2)]
        else:
            v = cast(int, v)
            conv1d = nn.Conv1d(in_channels, v, kernel_size=3, padding=1)
            attention = SolarAttention(num_channels=v)
            if batch_norm:
                layers += [conv1d, nn.BatchNorm1d(v), nn.ReLU(inplace=True),
                           attention]
            else:
                layers += [conv1d, nn.ReLU(inplace=True),
                           attention]
            in_channels = v
    return nn.Sequential(*layers)


class SolarAttention(nn.Module):
    def __init__(self, num_channels: int):
        super(SolarAttention, self).__init__()
        # model info
        self.num_channels = num_channels
        self.scale = num_channels ** -0.5

        # model layers
        self.q = nn.Conv1d(in_channels=num_channels,
                           out_channels=num_channels,
                           kernel_size=1)

        self.q = nn.Conv1d(in_channels=num_channels,
                           out_channels=num_channels,
                           kernel_size=1)

        self.k = nn.Conv1d(in_channels=num_channels,
                           out_channels=num_channels,
                           kernel_size=1)

        self.v = nn.Conv1d(in_channels=num_channels,
                           out_channels=num_channels,
                           kernel_size=1)

        self.attention_control = nn.Conv1d(in_channels=num_channels,
                                           out_channels=num_channels,
                                           kernel_size=1)

    def forward(self, x):
        """

        :param x: input with shape (batch_size, d, 320)
        :return:
        """
        # query, key and value projection
        q = self.q(x).transpose(1, 2)
        k = self.k(x)
        v = self.v(x).transpose(1, 2)

        # calculate scores for frames
        scores = torch.matmul(q, k) * self.scale
        weights = F.softmax(scores, dim=2)
        features = torch.matmul(weights, v).transpose(1, 2)
        # attention control
        features = self.attention_control(features)

        # combine x and features
        x = x + features
        return x


def sum_fcn_builder_attention(num_classes: int = 2):
    config = [256, 256, 'M', 256, 256, 'M', 256, 256, 256, 'M', 512, 512,
              512, 'M', 512, 512, 512, 'M', 'E', 'E']
    return SumFCNAttention(num_class=num_classes, cfg=config)
