import torch
import torch.nn as nn
import torch.nn.functional as F

def conv_dw(in_channels, out_channels, stride=1, dilation=1):
    """
    Depthwise separable convolution block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        dilation (int, optional): Dilation rate of the convolution. Defaults to 1.

    Returns:
        nn.Sequential: Depthwise separable convolution block.
    """
    layers = [
        nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=dilation, dilation=dilation, groups=in_channels, bias=False),
        nn.BatchNorm2d(in_channels),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]
    return nn.Sequential(*layers)

def conv(in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1, bn=True, relu=True):
    """
    Basic convolution block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel_size (int, optional): Kernel size of the convolution. Defaults to 3.
        stride (int, optional): Stride of the convolution. Defaults to 1.
        padding (int, optional): Padding of the convolution. Defaults to 1.
        dilation (int, optional): Dilation rate of the convolution. Defaults to 1.
        bn (bool, optional): Whether to include Batch Normalization. Defaults to True.
        relu (bool, optional): Whether to include ReLU activation. Defaults to True.

    Returns:
        nn.Sequential: Basic convolution block.
    """
    layers = [
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, bias=not bn),
    ]
    if bn:
        layers.append(nn.BatchNorm2d(out_channels))
    if relu:
        layers.append(nn.ReLU(inplace=True))
    return nn.Sequential(*layers)

def cpm_block(in_channels, out_channels):
    """
    Convolutional Pose Machine (CPM) block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        callable: Function representing the CPM block.
    """
    align = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
    trunk = nn.Sequential(
        conv_dw(out_channels, out_channels),
        conv_dw(out_channels, out_channels),
        conv_dw(out_channels, out_channels)
    )
    conv_layer = conv(out_channels, out_channels, bn=False)

    def forward(x):
        """
        Forward pass of the CPM block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        x = align(x)
        x = conv_layer(x + trunk(x))
        return x

    return forward

def initial_stage_block(num_channels, num_heatmaps, num_pafs):
    """
    Initial stage block.

    Args:
        num_channels (int): Number of input channels.
        num_heatmaps (int): Number of output heatmaps.
        num_pafs (int): Number of output Part Affinity Fields (PAFs).

    Returns:
        callable: Function representing the initial stage block.
    """
    trunk = nn.Sequential(
        conv(num_channels, num_channels, bn=False),
        conv(num_channels, num_channels, bn=False),
        conv(num_channels, num_channels, bn=False)
    )
    heatmaps = nn.Sequential(
        conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
        conv(512, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
    )
    pafs = nn.Sequential(
        conv(num_channels, 512, kernel_size=1, padding=0, bn=False),
        conv(512, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
    )

    def forward(x):
        """
        Forward pass of the initial stage block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: List containing output heatmaps and PAFs.
        """
        trunk_features = trunk(x)
        heatmaps_output = heatmaps(trunk_features)
        pafs_output = pafs(trunk_features)
        return [heatmaps_output, pafs_output]

    return forward

def refinement_stage_block(in_channels, out_channels):
    """
    Refinement stage block.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.

    Returns:
        callable: Function representing the refinement stage block.
    """
    initial = conv(in_channels, out_channels, kernel_size=1, padding=0, bn=False)
    trunk = nn.Sequential(
        conv(out_channels, out_channels),
        conv(out_channels, out_channels, dilation=2, padding=2)
    )

    def forward(x):
        """
        Forward pass of the refinement stage block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        initial_features = initial(x)
        trunk_features = trunk(initial_features)
        return initial_features + trunk_features

    return forward

def refinement_stage(num_channels, out_channels, num_heatmaps, num_pafs):
    """
    Refinement stage.

    Args:
        num_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        num_heatmaps (int): Number of output heatmaps.
        num_pafs (int): Number of output Part Affinity Fields (PAFs).

    Returns:
        callable: Function representing the refinement stage.
    """
    trunk = nn.Sequential(
        refinement_stage_block(num_channels, out_channels),
        refinement_stage_block(out_channels, out_channels),
        refinement_stage_block(out_channels, out_channels),
        refinement_stage_block(out_channels, out_channels),
        refinement_stage_block(out_channels, out_channels)
    )
    heatmaps = nn.Sequential(
        conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
        conv(out_channels, num_heatmaps, kernel_size=1, padding=0, bn=False, relu=False)
    )
    pafs = nn.Sequential(
        conv(out_channels, out_channels, kernel_size=1, padding=0, bn=False),
        conv(out_channels, num_pafs, kernel_size=1, padding=0, bn=False, relu=False)
    )

    def forward(x):
        """
        Forward pass of the refinement stage.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: List containing output heatmaps and PAFs.
        """
        trunk_features = trunk(x)
        heatmaps_output = heatmaps(trunk_features)
        pafs_output = pafs(trunk_features)
        return [heatmaps_output, pafs_output]

    return forward

def pose_estimation_with_mobilenet(num_refinement_stages=1, num_channels=128, num_heatmaps=19, num_pafs=38):
    """
    Pose Estimation with MobileNet model.

    Args:
        num_refinement_stages (int, optional): Number of refinement stages. Defaults to 1.
        num_channels (int, optional): Number of intermediate channels. Defaults to 128.
        num_heatmaps (int, optional): Number of output heatmaps. Defaults to 19.
        num_pafs (int, optional): Number of output Part Affinity Fields (PAFs). Defaults to 38.

    Returns:
        callable: Function representing the entire model.
    """
    model = nn.Sequential(
        conv(3, 32, stride=2, bn=False),
        conv_dw(32, 64),
        conv_dw(64, 128, stride=2),
        conv_dw(128, 128),
        conv_dw(128, 256, stride=2),
        conv_dw(256, 256),
        conv_dw(256, 512),
        conv_dw(512, 512, dilation=2, padding=2),
        conv_dw(512, 512),
        conv_dw(512, 512),
        conv_dw(512, 512),
        conv_dw(512, 512)
    )
    cpm = cpm_block(512, num_channels)

    initial_stage = initial_stage_block(num_channels, num_heatmaps, num_pafs)
    refinement_stages = nn.ModuleList()

    for idx in range(num_refinement_stages):
        refinement_stages.append(refinement_stage(num_channels + num_heatmaps + num_pafs, num_channels,
                                                  num_heatmaps, num_pafs))

    def forward(x):
        """
        Forward pass of the entire model.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            List[torch.Tensor]: List containing output heatmaps and PAFs.
        """
        backbone_features = model(x)
        backbone_features = cpm(backbone_features)

        stages_output = initial_stage(backbone_features)
        for refinement_stage in refinement_stages:
            stages_output.extend(
                refinement_stage(torch.cat([backbone_features, stages_output[-2], stages_output[-1]], dim=1)))

        return stages_output

    return forward
