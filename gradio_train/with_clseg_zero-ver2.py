import gradio as gr
import torch
import torchvision
from PIL import Image
import numpy as np

# CLIPSeg
# https://huggingface.co/docs/transformers/model_doc/clipseg
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation
from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler
import requests

_TITLE = "We'll show some segment image."
_DESCRIPTION = '''
Tell us what you want converted to 3D.
'''

processor = CLIPSegProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
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
    resizing = torchvision.transforms.Resize(np.array(images).shape[:2])
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits; preds = preds.unsqueeze(0) if len(preds.shape)<=2 else preds; print(preds.shape, resizing)
    preds = torch.sigmoid(resizing(preds))
    THRESHOLD = 0.5
    preds_thresholded = (preds[0] > THRESHOLD).int().numpy()
    segmented_image = np.where(preds_thresholded[..., np.newaxis] == 1, images, 255)
    segmented_image = Image.fromarray(segmented_image)
    result = pipeline(segmented_image).images[0]
    result = np.array(result)    
    
    lower_bound = np.array([175, 175, 175])
    upper_bound = np.array([215, 215, 215])

    new_color = np.array([255, 255, 255])

    result_white = np.where(np.all((result >= lower_bound) & (result <= upper_bound), axis=-1)[..., None], new_color, result)

    zero_ = pipeline(segmented_image).images[0]
    
    html = (
            "<div >"
            "<img  src='logo.png' alt='image One'>"
            + "</div>"
    )
    
    return segmented_image, zero_, html

# def inference(text):
#     html = (
#             "<div >"
#             "<img  src='logo.png' alt='image One'>"
#             + "</div>"
#     )
#     return html


# image_path = "logo.png"
# image_html = f'<img src="{image_path}" style="position:absolute; top:50px; left:50px; width:100px; height:100px;">'

iface = gr.Interface(
    fn=with_interface,
    inputs=[
        gr.Image(label="Upload image"),
        gr.Textbox(placeholder="Enter text descriptions", label="Texts"),
    ],
    outputs=[
        gr.Image(label="Segmentation Output"), # 352, 352
        gr.Image(label="multiview Output"), # 480, 320
        "html"
    ],
    title=_TITLE,
    description=_DESCRIPTION,
)

iface.launch(debug=True, share=True)