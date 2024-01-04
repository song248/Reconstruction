import numpy as np
import gradio as gr


def sepia(input_img):
    sep_fil = np.array([
       [0.393, 0.769, 0.189],
       [0.349, 0.686, 0.168],
       [0.272, 0.534, 0.131] 
    ])
    sep_img = input_img.dot(sep_fil.T)
    sep_img /= sep_img.max()

    return sep_img

demo = gr.Interface(fn=sepia, inputs='image', outputs='image')

demo.launch(debug=True, share=True)