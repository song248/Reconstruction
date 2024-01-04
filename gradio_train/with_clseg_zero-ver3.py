import os
import gradio as gr
import torch
import torchvision
from PIL import Image
import numpy as np
import cv2

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


def insert_images(background_path, overlay_paths):
    
    positions = [(95, 175), (95, 398), (440, 180), (440, 330), (630, 180), (630, 330)]
    widths = [273, 273, 175, 175, 175, 175]
    
    background = Image.open(background_path)
    for overlay_path, position, width in zip(overlay_paths, positions, widths):
        overlay = Image.open(overlay_path)
        height = int(overlay.size[1] * (width / overlay.size[0]))
        overlay.thumbnail((width, height))

        if overlay.mode in ('RGBA', 'LA') or (overlay.mode == 'P' and 'transparency' in overlay.info):
            mask = overlay.convert('L')
        else:
            mask = None
        background.paste(overlay, position, mask)
    background.save("C:/Users/lab/Desktop/gradio_train/image/output_image.jpg")
    # background.save("C:/User/lab/Documents/G-Drive/output_image.jpg")


def photo_print(origin, seg_men, six_p):
    save_dir = "C:/Users/lab/Desktop/gradio_train/image"
    base_filename = "image_"
    base_filename2 = "segment_"
    
    o_file_path = save_dir+'/origin.png'
    # cv2.imwrite(o_file_path, origin)
    ori_ = Image.fromarray(origin) 
    ori_.save(o_file_path)
    
    s_file_path =  save_dir+'/segment_1.png'
    print('  -- - - - - --------------  ')
    print(type(seg_men))
    seg_men.save(s_file_path)
    
    overlay_paths = [o_file_path, s_file_path]
    
    for i, image in enumerate(six_p):
        filename = f"{base_filename}{i + 1}.png"
        file_path = os.path.join(save_dir, filename)
        image.save(file_path)
        if i == 0 or i == 1 or i == 4 or i == 5:
            overlay_paths.append(file_path)
    
    background_path = save_dir+'/horizontal.png'
    
    insert_images(background_path, overlay_paths)
    

def with_interface(images, texts):
    o_file_path = "C:/Users/lab/Desktop/gradio_train/image/origin.png"
    cv2.imwrite(o_file_path, images)
    
    resizing = torchvision.transforms.Resize(np.array(images).shape[:2])
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
    preds = outputs.logits; preds = preds.unsqueeze(0) if len(preds.shape)<=2 else preds; print(preds.shape, resizing)
    preds = torch.sigmoid(resizing(preds))
    THRESHOLD = 0.5
    preds_thresholded = (preds[0] > THRESHOLD).int().numpy()
    segmented_image = np.where(preds_thresholded[..., np.newaxis] == 1, images, 255)
    s_file_path = "C:/User/lab/Documents/G-Drive/segment_1.png"
    cv2.imwrite(s_file_path, segmented_image)
    segmented_image = Image.fromarray(segmented_image)
    result = pipeline(segmented_image).images[0]
    result = np.array(result)    
    
    lower_bound = np.array([175, 175, 175])
    upper_bound = np.array([215, 215, 215])

    new_color = np.array([255, 255, 255])

    result_white = np.where(np.all((result >= lower_bound) & (result <= upper_bound), axis=-1)[..., None], new_color, result)

    h, w, _= result_white.shape
    mini_h = h//3; mini_w = w//2
    a1 = Image.fromarray(result_white[0:mini_h, 0:mini_w].astype('uint8')); a2 = Image.fromarray(result_white[mini_h:mini_h*2, 0:mini_w].astype('uint8')); a3 = Image.fromarray(result_white[mini_h*2:mini_h*3, 0:mini_w].astype('uint8'));
    a4 = Image.fromarray(result_white[0:mini_h, mini_w:mini_w*2].astype('uint8')); a5 = Image.fromarray(result_white[mini_h:mini_h*2, mini_w:mini_w*2].astype('uint8')); a6 = Image.fromarray(result_white[mini_h*2:mini_h*3, mini_w:mini_w*2].astype('uint8'));
    
    image_list = [a1, a2, a3, a4, a5, a6]
    photo_print(images, segmented_image, image_list)
                                                                                                                                                                                              
    zero_ = pipeline(segmented_image).images[0]
    
    html = (
            "<div >"
            "<img  src='logo.png' alt='image One'>"
            + "</div>"
    )
    
    
    
    return segmented_image, zero_, html

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