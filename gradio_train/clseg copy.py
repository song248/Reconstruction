import gradio as gr
import torch
from PIL import Image

# CLIPSeg
# https://huggingface.co/docs/transformers/model_doc/clipseg
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image
import requests

processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


def clipseg_interface(texts, images):
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits)
    segment_output = (probabilities * 2 - 1).detach().numpy()
    return segment_output

iface = gr.Interface(
    fn=clipseg_interface,
    inputs=[
        gr.Textbox(placeholder="Enter text descriptions", label="Texts"),
        gr.Image(label="Upload image"),
    ],
    outputs=gr.Image(label="Segmentation Output"),
)

iface.launch(debug=True, share=True)