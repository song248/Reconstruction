import gradio as gr
import torch
from PIL import Image
import numpy as np

# CLIPSeg
# https://huggingface.co/docs/transformers/model_doc/clipseg
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import requests

# processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
# model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)
# Feel free to tune the scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to('cuda:0')

def run_diffusion_pipeline(input_image):
    # image_bytes = input_image.read()
    cond = Image.fromarray(input_image)
    result = pipeline(cond).images[0]
    return result

iface = gr.Interface(
    fn=run_diffusion_pipeline,
    inputs=gr.Image(label="Upload image"),
    outputs=gr.Image(label="Segmentation Output"),
)

iface.launch(debug=True, share=True)