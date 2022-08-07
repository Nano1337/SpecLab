from xml.dom.minidom import Element
import gradio as gr
from ASPP import SRDetectModel
import torch
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import matplotlib.pyplot as plt

def speclab(img):

    # initialize the model
    model = SRDetectModel()
    state_dict = torch.load(r"C:\Users\haoli\OneDrive\Documents\SpecLab\logs\model_checkpoints\epoch_009.ckpt")["state_dict"]
    for key in list(state_dict.keys()):
        state_dict[key.replace('net.', '')] = state_dict.pop(key)
    model.load_state_dict(state_dict)
    model = model.to("cpu")
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
    img[output==255] = 0

    return img

# define app features and run

title = "SpecLab Demo"
description = "<p style='text-align: center'>Gradio demo for an ASPP model architecture trained on the SpecLab dataset. To use it, simply add your image, or click one of the examples to load them. </p>"
article = "<p style='text-align: center'><a href='https://github.com/Nano1337/SpecLab'>Github Repo</a></p>"
examples = [
    [r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_img\00000.png"]
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