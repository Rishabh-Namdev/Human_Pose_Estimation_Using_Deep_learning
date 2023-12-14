import torch.nn as nn

def find_parameters(model, condition):
    """
    Find parameters in a PyTorch model based on a given condition.

    Args:
        model (nn.Module): PyTorch model.
        condition (function): A condition function to filter parameters.

    Returns:
        generator: Generator yielding parameters based on the condition.
    """
    for module in model.modules():
        for param_name, param in module.named_parameters():
            if condition(module, param_name):
                yield param

def is_standard_conv(module, param_name):
    """
    Check if a parameter corresponds to a standard convolution layer.

    Args:
        module (nn.Module): PyTorch module.
        param_name (str): Name of the parameter.

    Returns:
        bool: True if the parameter corresponds to a standard convolution layer, False otherwise.
    """
    return isinstance(module, nn.Conv2d) and module.groups == 1 and param_name == 'weight'

def is_depthwise_conv(module, param_name):
    """
    Check if a parameter corresponds to a depthwise convolution layer.

    Args:
        module (nn.Module): PyTorch module.
        param_name (str): Name of the parameter.

    Returns:
        bool: True if the parameter corresponds to a depthwise convolution layer, False otherwise.
    """
    return isinstance(module, nn.Conv2d) and module.groups == module.in_channels and module.in_channels == module.out_channels and param_name == 'weight'

def is_batch_norm(module, param_name):
    """
    Check if a parameter corresponds to a batch normalization layer.

    Args:
        module (nn.Module): PyTorch module.
        param_name (str): Name of the parameter.

    Returns:
        bool: True if the parameter corresponds to a batch normalization layer, False otherwise.
    """
    return isinstance(module, nn.BatchNorm2d) and param_name in ['weight', 'bias']
