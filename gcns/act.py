import torch.nn as nn

def build_activation_layer(act_cfg):
    """Construct the activation layer based on the act_cfg string."""
    if isinstance(act_cfg, dict):
        act_type = act_cfg.get('type', 'ReLU')
    else:
        act_type = act_cfg

    if act_type == 'ReLU':
        return nn.ReLU()
    elif act_type == 'LeakyReLU':
        return nn.LeakyReLU()
    elif act_type == 'Sigmoid':
        return nn.Sigmoid()
    elif act_type == 'Tanh':
        return nn.Tanh()
    else:
        return None
