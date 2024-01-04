import gradio as gr
import torch
from PIL import Image

# CLIPSeg
# https://huggingface.co/docs/transformers/model_doc/clipseg
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import requests

_TITLE = "We'll show some segment image."
_DESCRIPTION = '''
Tell us what you want converted to 3D.
'''

processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")

pipeline = DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.1", custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16
)
# Feel free to tune the scheduler
pipeline.scheduler = EulerAncestralDiscreteScheduler.from_config(
    pipeline.scheduler.config, timestep_spacing='trailing'
)
pipeline.to('cuda:0')

def with_interface(images, texts):
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits)
    seg_ = (probabilities * 2 - 1).detach().numpy()
    
    cond = Image.fromarray(images)
    zero_ = pipeline(cond).images[0]
    return seg_, zero_

css = ".output_image {height: 300px !important; width: 60% !important;}"

iface = gr.Interface(
    fn=with_interface,
    inputs=[
        gr.Image(label="Upload image"),
        gr.Textbox(placeholder="Enter text descriptions", label="Texts"),
    ],
    outputs=[
        gr.Image(label="Segmentation Output"), # 352, 352
        gr.Image(label="multiview Output"), # 480, 320
    ],
    css=css,
    title=_TITLE,
    description=_DESCRIPTION,
)

iface.launch(debug=True, share=True)