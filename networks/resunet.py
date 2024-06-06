import torch.nn as nn
import torch


class ConvolutionBatchNormblock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        adaptive convolution + batch norm + activation block
        """

        # convolutional layer

        self.conv1 = nn.Conv2d(in_channels=in_channels,
                               out_channels=out_channels, padding=padding, kernel_size=kernel_size, stride=stride)

        # batch norm
        self.bn = nn.BatchNorm2d(num_features=out_channels)

        # activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv1(x)

        return x


class DownSampleBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, first=False):
        # conv block

        self.first = first

        if first == True:
            # initialize first time convolution

            self.single_first_conv = nn.Conv2d(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding="same", stride=1)

            self.first_conv_block = ConvolutionBatchNormblock(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)

        else:
            self.conv_block_1 = ConvolutionBatchNormblock(
                in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=2, padding=0)

            self.conv_block_2 = ConvolutionBatchNormblock(
                in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1, padding="same")

        # activation function
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # pass through convolutional block

        if self.first:
            x_ = self.single_first_conv(x)
            x_ = self.first_conv_block(x_)

            x = x_ + x

            x = self.relu(x)

            return x

        else:
            x_ = self.conv_block_1(x)
            x_ = self.conv_block_2(x_)

            x = x_ + x

            x = self.relu(x)

            return x


class UpSampleSubBlock(nn.Module):
    def __init__(self, in_channels, out_channels_list: list, kernel_size):

        # upsampling feature
        self.up = nn.ConvTranspose2d(
            in_channels=in_channels, out_channels=out_channels_list[0], stride=2, kernel_size=kernel_size, padding=0)

        # convolutional blocks
        self.conv_block_1 = ConvolutionBatchNormblock(
            in_channels=in_channels, out_channels=out_channels_list[0], kernel_size=kernel_size, stride=1, padding="same")

        self.conv_block_2 = ConvolutionBatchNormblock(
            in_channels=out_channels_list[0], out_channels=out_channels_list[1], kernel_size=kernel_size, stride=1, padding="same")

        # activation
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, s):
        # upsample original path
        x = self.up(x)

        x = torch.concat(x, s, dim=1)

        x_ = self.conv_block_1(x)
        x_ = self.conv_block_2(x_)

        x = x + x_

        x = self.relu(x)

        return x


class ResUNet(nn.Module):
    def __init__(self, color_channels, num_classes):

        # downward path
        model_kernel_size = 3

        self.d1 = DownSampleBlock(
            in_channels=color_channels, out_channels=64, kernel_size=model_kernel_size, first=True)

        self.d2 = DownSampleBlock(
            in_channels=64, out_channels=128, kernel_size=model_kernel_size)

        self.d3 = DownSampleBlock(
            in_channels=64, out_channels=128, kernel_size=model_kernel_size)

        # bridge
        self.b1 = DownSampleBlock(
            in_channels=64, out_channels=128, kernel_size=model_kernel_size)
