import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def delete_zero(tensor):
    zero = torch.zeros_like(tensor)
    index = tensor == zero
    return tensor[~index].unsqueeze(0)

def mean_activation(tensor):
    tensor_delete = delete_zero(tensor)
    mean_value = tensor_delete.mean(dim=1, keepdim=True)
    tensor_hat = tensor - mean_value
    active_tensor = torch.nn.Sigmoid()(tensor_hat)
    return active_tensor


class SRMLayer(nn.Module):
    def __init__(self, channel):
        super(SRMLayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // 16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // 16, channel, bias=False),
            nn.Sigmoid()
        )
        self.conv1d = nn.Conv1d(in_channels=channel,out_channels=channel,kernel_size=2,stride=1)
        self.activation = nn.Sigmoid()

    def channel_attention(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

    def _style_pooling(self, x, eps=1e-5):
        N, C, _, _ = x.size()
        channel_mean = x.view(N, C, -1).mean(dim=2, keepdim=True)   # N x C x 1
        channel_var = x.view(N, C, -1).var(dim=2, keepdim=True) + eps   # N x C x 1
        channel_std = channel_var.sqrt()
        t = torch.cat((channel_mean, channel_std), dim=2)   # N x C x 2
        return t

    def _style_integration(self, t):
        z = self.conv1d(t)   # N x C x 1
        z_hat = z.unsqueeze(2)   # N x C x 1 x 1
        g = self.activation(z_hat)  # N x C x 1 x 1
        return g

    def forward(self, x):
        # N x C x H x W
        se = self.channel_attention(x)
        # N x C x 2
        t = self._style_pooling(se)
        # N x C x 1 x 1
        g = self._style_integration(t)
        return x * g


class SRMConvBlock(nn.Module):
    def __init__(self, in_features):
        super(SRMConvBlock, self).__init__()

        conv_block = [nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features),
                      nn.ReLU(inplace=True),
                      nn.ReflectionPad2d(1),
                      nn.Conv2d(in_features, in_features, 3),
                      nn.InstanceNorm2d(in_features)]

        self.conv_block = nn.Sequential(*conv_block)
        self.srm_layer = SRMLayer(in_features)


    def forward(self, x):
        t = x
        t_conv = self.conv_block(t)
        r = t_conv
        r_srm = self.srm_layer(r)
        return x + r_srm


class SRMResidualLayer(nn.Module):
    def __init__(self, in_features):
        super(SRMResidualLayer, self).__init__()
        self.srm_layer = SRMLayer(in_features)

    def forward(self, x):
        t = x
        r_srm = self.srm_layer(t)
        return x + r_srm


class Generator(nn.Module):
    def __init__(self, input_nc, output_nc, n_residual_blocks=9):
        super(Generator, self).__init__()

        # Initial convolution block
        model = [   nn.ReflectionPad2d(3),
                    nn.Conv2d(input_nc, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True) ]

        # Downsampling
        in_features = 64
        out_features = in_features*2
        for _ in range(2):
            model += [  nn.Conv2d(in_features, out_features, 3, stride=2, padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features*2

        # Residual blocks
        for _ in range(n_residual_blocks):
            model += [SRMConvBlock(in_features)]

        # Upsampling
        out_features = in_features//2
        for _ in range(2):
            model += [  nn.ConvTranspose2d(in_features, out_features, 3, stride=2, padding=1, output_padding=1),
                        nn.InstanceNorm2d(out_features),
                        nn.ReLU(inplace=True) ]
            in_features = out_features
            out_features = in_features//2

        # Output layer
        model += [  nn.ReflectionPad2d(3),
                    nn.Conv2d(64, output_nc, 7),
                    nn.Tanh() ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        return self.model(x)


class StyleDiscriminator(nn.Module):
    def __init__(self, input_nc):
        super(StyleDiscriminator, self).__init__()


        model = [   nn.Conv2d(input_nc, 64, 4, stride=2, padding=1),
                    SRMResidualLayer(64),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(64, 128, 4, stride=2, padding=1),
                    SRMResidualLayer(128),
                    nn.InstanceNorm2d(128),
                    nn.LeakyReLU(0.2, inplace=True) ]

        model += [  nn.Conv2d(128, 256, 4, stride=2, padding=1),
                    SRMResidualLayer(256),
                    nn.InstanceNorm2d(256),
                    nn.LeakyReLU(0.2, inplace=True) ]

        self.encoder_layers_SD = nn.Sequential(*model)

        model += [nn.Conv2d(256, 1, kernel_size=1, stride=1, padding=0, bias=False)]

        self.encoder_layers_D = nn.Sequential(*model)

        self.avg_pool1 = nn.AdaptiveAvgPool2d(8)
        self.avg_pool2 = nn.AdaptiveAvgPool2d(1)

        self.maxpool1 = nn.AdaptiveMaxPool2d(8)
        self.maxpool2 = nn.AdaptiveMaxPool2d(1)


    def forward(self, x):

        # For Decision
        x_copy = x
        x_copy = self.encoder_layers_D(x_copy)
        scalar_D = F.avg_pool2d(x_copy, x_copy.size()[2:]).view(x_copy.size()[0], -1)

        # For Style Vector
        x =  self.encoder_layers_SD(x)
        y = x
        x1 = self.avg_pool1(x)
        y1 = self.maxpool1(y)
        combine1 = x1 + y1
        x2 = self.avg_pool2(combine1)
        y2 = self.maxpool2(combine1)
        x3 = x2.view(x.size()[0], -1)
        y3 = y2.view(y.size()[0], -1)
        z = x3 + y3

        z = torch.mm(z.t(),z)  # VT*V
        z = torch.triu(z, diagonal=1)
        z = z.view(1, -1)
        z = mean_activation(z)

        return scalar_D.squeeze(-1), z


