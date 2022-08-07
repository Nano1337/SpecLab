import torch
from demo.ASPP import SRDetectModel

def aspp():
    model = SRDetectModel()
    state_dict = torch.load(r"C:\Users\haoli\OneDrive\Documents\SpecLab\logs\model_checkpoints\epoch_009.ckpt")["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace('net.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to("cpu")
    return model



