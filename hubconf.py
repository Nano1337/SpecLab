dependencies = ['torch']

import torch
from demo.ASPP import SRDetectModel

import urllib
url, filename = ("https://github.com/Nano1337/SpecLab/blob/main/logs/model_checkpoints/epoch_009.ckpt?raw=true", "epoch_009.ckpt")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename) 

def srdetect():
    model = SRDetectModel()
    state_dict = torch.load("epoch_009.ckpt")["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace('net.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict, map_location=torch.device('cpu'))
    return model



