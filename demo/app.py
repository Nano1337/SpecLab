import gradio as gr
import torch
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import numpy as np

import urllib

# get image examples from github
url, filename = ("https://github.com/Nano1337/SpecLab/blob/main/examples/05829.png?raw=true", "05829.png") # make sure to use "copy image address"
url, filename = ("https://github.com/Nano1337/SpecLab/blob/main/examples/10384.png?raw=true", "10384.png")
try: urllib.URLopener().retrieve(url, filename)
except: urllib.request.urlretrieve(url, filename) 

def speclab(img):

    # initialize the model
    model = torch.hub.load('Nano1337/SpecLab', 'srdetect', force_reload=True) # for some reasons loads the model in src rather than demo
    model.eval()

    # preprocess image to be used as input
    transforms = A.Compose([
        A.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
    input = transforms(image=img)['image']
    input = input.unsqueeze(0)

    # model prediction
    output = model(input)

    # overlay output onto original image
    img[output==255] = [0, 255, 0]

    return img

# define app features and run
img = np.zeros((360, 640, 3), dtype=np.uint8)   
speclab(img)

title = "SpecLab Demo"
description = "<p style='text-align: center'>Gradio demo for an ASPP model architecture trained on the SpecLab dataset. To use it, simply add your image, or click one of the examples to load them. </p>"
article = "<p style='text-align: center'><a href='https://github.com/Nano1337/SpecLab'>Github Repo</a></p>"
examples = [ # need to manually delete cache everytime new examples are added
    ['05829.png'], 
    ["10384.png"]
]
css = "#0 {object-fit: contain;} #1 {object-fit: contain;}"
demo = gr.Interface(fn=speclab, 
                    title=title, 
                    description=description,
                    article=article,
                    inputs=gr.Image(elem_id=0, show_label=False), 
                    outputs=gr.Image(elem_id=1, show_label=False),
                    css=css, 
                    examples=examples, 
                    cache_examples=True,
                    allow_flagging='never')
demo.launch()

# make sure to connect external SSD to computer before using locally
