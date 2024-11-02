import torch.nn as nn

def build_norm_layer(norm_cfg, out_channels):
    """Construct the normalization layer based on the norm_cfg string."""
    
    if isinstance(norm_cfg, dict):
        norm_type = norm_cfg.get('type', 'BN')
    else:
        norm_type = norm_cfg

    if norm_type == 'BN':
        return ('bn', nn.BatchNorm2d(out_channels))
    elif norm_type == 'LN':
        return ('ln', nn.LayerNorm(out_channels))
    elif norm_type == 'IN':
        return ('in', nn.InstanceNorm2d(out_channels))
    else:
        return (None, None)  # 如果不需要正則化層，返回 (None, None)