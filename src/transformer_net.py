# -*- coding: utf-8 -*-


import torch
from torch.nn import functional as F


class TransformerNet(torch.nn.Module):

    def __init__(self):
        super(TransformerNet, self).__init__()
        # Initial convolution layers
        self.conv1 = ConvLayer(3, 32, kernel_size=9, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(32, affine=True)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, stride=2)
        self.in2 = torch.nn.InstanceNorm2d(64, affine=True)
        self.conv3 = ConvLayer(64, 128, kernel_size=3, stride=2)
        self.in3 = torch.nn.InstanceNorm2d(128, affine=True)
        # Residual layers
        self.res1 = ResidualBlock(128)
        self.res2 = ResidualBlock(128)
        self.res3 = ResidualBlock(128)
        self.res4 = ResidualBlock(128)
        self.res5 = ResidualBlock(128)
        # Upsampling Layers
        self.deconv1 = UpsampleConvLayer(128, 64, kernel_size=3, stride=1, upsample=2)
        self.in4 = torch.nn.InstanceNorm2d(64, affine=True)
        self.deconv2 = UpsampleConvLayer(64, 32, kernel_size=3, stride=1, upsample=2)
        self.in5 = torch.nn.InstanceNorm2d(32, affine=True)
        self.deconv3 = ConvLayer(32, 3, kernel_size=9, stride=1)
        # Non-linearities
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def forward(self, X, in_weights=None):
        if in_weights is None:
            y = self.relu(self.in1(self.conv1(X)))
            y = self.relu(self.in2(self.conv2(y)))
            y = self.relu(self.in3(self.conv3(y)))
            y = self.res1(y)
            y = self.res2(y)
            y = self.res3(y)
            y = self.res4(y)
            y = self.res5(y)
            y = self.relu(self.in4(self.deconv1(y)))
            y = self.relu(self.in5(self.deconv2(y)))
            y = self.deconv3(y)
            y = self.sigmoid(y)
        else:
            y = self.conv1(X)
            y = F.instance_norm(y, weight=in_weights['in1.weight'], bias=in_weights['in1.bias'])
            y = self.relu(y)
            y = self.conv2(y)
            y = F.instance_norm(y, weight=in_weights['in2.weight'], bias=in_weights['in2.bias'])
            y = self.relu(y)
            y = self.conv3(y)
            y = F.instance_norm(y, weight=in_weights['in3.weight'], bias=in_weights['in3.bias'])
            y = self.relu(y)
            y = self.res1(y, {"in1.weight": in_weights['res1.in1.weight'], "in1.bias": in_weights['res1.in1.bias'],
                              "in2.weight": in_weights['res1.in2.weight'], "in2.bias": in_weights['res1.in2.bias']})
            y = self.res2(y, {"in1.weight": in_weights['res2.in1.weight'], "in1.bias": in_weights['res2.in1.bias'],
                              "in2.weight": in_weights['res2.in2.weight'], "in2.bias": in_weights['res2.in2.bias']})
            y = self.res3(y, {"in1.weight": in_weights['res3.in1.weight'], "in1.bias": in_weights['res3.in1.bias'],
                              "in2.weight": in_weights['res3.in2.weight'], "in2.bias": in_weights['res3.in2.bias']})
            y = self.res4(y, {"in1.weight": in_weights['res4.in1.weight'], "in1.bias": in_weights['res4.in1.bias'],
                              "in2.weight": in_weights['res4.in2.weight'], "in2.bias": in_weights['res4.in2.bias']})
            y = self.res5(y, {"in1.weight": in_weights['res5.in1.weight'], "in1.bias": in_weights['res5.in1.bias'],
                              "in2.weight": in_weights['res5.in2.weight'], "in2.bias": in_weights['res5.in2.bias']})
            y = self.deconv1(y)
            y = F.instance_norm(y, weight=in_weights['in4.weight'], bias=in_weights['in4.bias'])
            y = self.relu(y)
            y = self.deconv2(y)
            y = F.instance_norm(y, weight=in_weights['in5.weight'], bias=in_weights['in5.bias'])
            y = self.relu(y)
            y = self.deconv3(y)
            y = self.sigmoid(y)
        return y


class ConvLayer(torch.nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvLayer, self).__init__()
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        out = self.reflection_pad(x)
        out = self.conv2d(out)
        return out


class ResidualBlock(torch.nn.Module):

    """ResidualBlock
    introduced in: https://arxiv.org/abs/1512.03385
    recommended architecture: http://torch.ch/blog/2016/02/04/resnets.html
    """

    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in1 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.conv2 = ConvLayer(channels, channels, kernel_size=3, stride=1)
        self.in2 = torch.nn.InstanceNorm2d(channels, affine=True)
        self.relu = torch.nn.ReLU()

    def forward(self, x, in_weights=None):
        residual = x
        if in_weights is None:
            out = self.relu(self.in1(self.conv1(x)))
            out = self.in2(self.conv2(out))
        else:
            out = self.conv1(x)
            out = F.instance_norm(out, weight=in_weights['in1.weight'], bias=in_weights['in1.bias'])
            out = self.relu(out)
            out = self.conv2(out)
            out = F.instance_norm(out, weight=in_weights['in2.weight'], bias=in_weights['in2.bias'])
        out = out + residual
        return out


class UpsampleConvLayer(torch.nn.Module):

    """UpsampleConvLayer
    Upsamples the input and then does a convolution. This method gives better results
    compared to ConvTranspose2d.
    ref: http://distill.pub/2016/deconv-checkerboard/
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None):
        super(UpsampleConvLayer, self).__init__()
        self.upsample = upsample
        if upsample:
            self.upsample_layer = torch.nn.Upsample(mode='nearest', scale_factor=upsample)
        reflection_padding = kernel_size // 2
        self.reflection_pad = torch.nn.ReflectionPad2d(reflection_padding)
        self.conv2d = torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride)

    def forward(self, x):
        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)
        return out
