import collections

def load_model_state(model, pretrained_checkpoint):
    """
    Load the state of a PyTorch model from a pre-trained checkpoint.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to load the state into.
    - pretrained_checkpoint (dict): The pre-trained checkpoint containing the state.

    Returns:
    None
    """
    model_state = model.state_dict()
    pretrained_state = pretrained_checkpoint['state_dict']
    new_model_state = collections.OrderedDict()

    for key, value in model_state.items():
        if key in pretrained_state and pretrained_state[key].size() == value.size():
            new_model_state[key] = pretrained_state[key]
        else:
            new_model_state[key] = value
            print('[WARNING] No pre-trained parameters found for {}'.format(key))

    model.load_state_dict(new_model_state)


def load_from_mobilenet(model, pretrained_checkpoint):
    """
    Load the state of a PyTorch model from a MobileNet pre-trained checkpoint.

    Parameters:
    - model (torch.nn.Module): The PyTorch model to load the state into.
    - pretrained_checkpoint (dict): The MobileNet pre-trained checkpoint containing the state.

    Returns:
    None
    """
    model_state = model.state_dict()
    pretrained_state = pretrained_checkpoint['state_dict']
    new_model_state = collections.OrderedDict()

    for key, value in model_state.items():
        modified_key = key.replace('model', 'module.model') if 'model' in key else key
        if modified_key in pretrained_state and pretrained_state[modified_key].size() == value.size():
            new_model_state[key] = pretrained_state[modified_key]
        else:
            new_model_state[key] = value
            print('[WARNING] No pre-trained parameters found for {}'.format(key))

    model.load_state_dict(new_model_state)
