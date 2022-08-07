import gradio as gr

with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            text1 = gr.Textbox(label="prompt 1")
            text2 = gr.Textbox(label="prompt 2")
        with gr.Column():
            img1 = gr.Image(r"D:\GLENDA_v1.5_no_pathology\no_pathology\GLENDA_img\00000.png")
            btn = gr.Button("Go").style(full_width=True)
demo.launch()