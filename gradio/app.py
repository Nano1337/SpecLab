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

css = ".output-image, .input-image {height: 30rem !important; width: 100% !important;}"
examples = [
    r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_img\00000.png"
]
demo = gr.Interface(fn=speclab, 
                    inputs=gr.Image(), 
                    outputs=gr.Image(shape=(360, 640)),
                    css=css, 
                    examples=examples)
demo.launch()