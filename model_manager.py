import torch
from models.transform_net import TransformNet

_model_cache = {}

def get_model(model_path,device):
    key = (model_path,str(device))

    if key in _model_cache:
        return _model_cache[key]
    
    model = TransformNet().to(device)
    model.load_state_dict(                             # loads trained weights
        torch.load(
            model_path,
            map_location = device
        )
    )
    model.eval()                                       # switching to inference mode instead of trianing mode

    _model_cache[key] = model

    return model