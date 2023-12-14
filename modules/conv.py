import torch
import torch.nn as nn

def convolution_block(in_channels, out_channels, kernel_size=3, padding=1, use_batch_norm=True, dilation=1, stride=1, use_relu=True, use_bias=True):
    """
    Create a convolutional block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size. Defaults to 1.
        use_batch_norm (bool, optional): Use Batch Normalization. Defaults to True.
        dilation (int, optional): Dilation factor for the convolution. Defaults to 1.
        stride (int, optional): Stride for the convolution. Defaults to 1.
        use_relu (bool, optional): Use ReLU activation. Defaults to True.
        use_bias (bool, optional): Use bias in the convolution. Defaults to True.

    Returns:
        nn.Module: Sequential convolutional block.
    """
    conv_layer = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation=dilation, bias=use_bias)
    layers = [conv_layer]

    if use_batch_norm:
        layers.append(nn.BatchNorm2d(out_channels))

    if use_relu:
        layers.append(nn.ReLU(inplace=True))

    return nn.Sequential(*layers)

def depthwise_separable_conv_block(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1, use_batch_norm=True):
    """
    Create a depthwise separable convolution block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size. Defaults to 1.
        stride (int, optional): Stride for the convolution. Defaults to 1.
        dilation (int, optional): Dilation factor for the convolution. Defaults to 1.
        use_batch_norm (bool, optional): Use Batch Normalization. Defaults to True.

    Returns:
        nn.Module: Sequential depthwise separable convolution block.
    """
    layers = [
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
    ]

    if use_batch_norm:
        layers.append(nn.BatchNorm2d(in_channels))

    layers.append(nn.ReLU(inplace=True))

    layers.extend([
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ])

    return nn.Sequential(*layers)

def depthwise_separable_conv_no_bn(in_channels, out_channels, kernel_size=3, padding=1, stride=1, dilation=1):
    """
    Create a depthwise separable convolution block without Batch Normalization.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Size of the convolutional kernel. Defaults to 3.
        padding (int, optional): Padding size. Defaults to 1.
        stride (int, optional): Stride for the convolution. Defaults to 1.
        dilation (int, optional): Dilation factor for the convolution. Defaults to 1.

    Returns:
        nn.Module: Sequential depthwise separable convolution block without Batch Normalization.
    """
    layers = [
        nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation=dilation, groups=in_channels, bias=False),
        nn.ELU(inplace=True),
        nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
        nn.ELU(inplace=True),
    ]

    return nn.Sequential(*layers)
