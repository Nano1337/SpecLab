dependencies = ['torch']

import torch
from demo.ASPP import SRDetectModel

def srdetect():
    model = SRDetectModel()
    checkpoint = torch.hub.load_state_dict_from_url('https://github.com/Nano1337/SpecLab/blob/main/logs/model_checkpoints/epoch_009.ckpt?raw=true', map_location="cpu")
    state_dict = {key.replace("net.", ""): value for key, value in checkpoint["state_dict"].items()}
    model.load_state_dict(state_dict)
    return model



