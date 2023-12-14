def calculate_l2_loss(input_tensor, target_tensor, mask_tensor, batch_size):
    """
    Calculate L2 loss between the input and target tensors, masked by a provided mask tensor.

    Parameters:
    - input_tensor (torch.Tensor): The input tensor.
    - target_tensor (torch.Tensor): The target tensor.
    - mask_tensor (torch.Tensor): The mask tensor used for masking the loss computation.
    - batch_size (int): The size of the batch.

    Returns:
    torch.Tensor: The L2 loss.
    """
    loss = (input_tensor - target_tensor) * mask_tensor
    loss = (loss * loss) / (2.0 * batch_size)

    return loss.sum()
