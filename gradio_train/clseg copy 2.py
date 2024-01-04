import gradio as gr
import torch
from PIL import Image
import numpy as np
from io import BytesIO
import base64

# CLIPSeg
# https://huggingface.co/docs/transformers/model_doc/clipseg
from transformers import AutoProcessor, CLIPSegForImageSegmentation
from PIL import Image

processor = AutoProcessor.from_pretrained("CIDAS/clipseg-rd64-refined")
model = CLIPSegForImageSegmentation.from_pretrained("CIDAS/clipseg-rd64-refined")


def clipseg_interface(images, texts):
    inputs = processor(text=texts, images=images, padding=True, return_tensors="pt")
    outputs = model(**inputs)
    logits = outputs.logits
    probabilities = torch.sigmoid(logits)
    segment_output = (probabilities * 2 - 1).detach().numpy()
    return segment_output


def display_segmentation_on_image(texts, images):
    segmentation_output = clipseg_interface(texts, images)

    # 입력 이미지 표시
    input_image = Image.fromarray((images[0] * 255).astype(np.uint8))
    input_image_byte_array = BytesIO()
    input_image.save(input_image_byte_array, format="PNG")
    input_image_base64 = base64.b64encode(input_image_byte_array.getvalue()).decode("utf-8")
    input_image_html = f'<img src="data:image/png;base64,{input_image_base64}" alt="input_image" style="width:400px;height:400px;">'

    # 세그멘테이션 결과 표시
    segmentation_image = Image.fromarray(((segmentation_output[0] + 1) / 2 * 255).astype(np.uint8))  # 정규화 해제
    segmentation_image_byte_array = BytesIO()
    segmentation_image.save(segmentation_image_byte_array, format="PNG")
    segmentation_image_base64 = base64.b64encode(segmentation_image_byte_array.getvalue()).decode("utf-8")
    segmentation_image_html = f'<img src="data:image/png;base64,{segmentation_image_base64}" alt="segmentation_image" style="width:400px;height:400px;">'

    # HTML 및 스타일링을 사용하여 입력 및 세그멘테이션 결과를 가운데 정렬
    result_html = f"<div style='display: flex; justify-content: center;'><div>{input_image_html}</div><div>{segmentation_image_html}</div></div>"

    return result_html

iface = gr.Interface(
    fn=display_segmentation_on_image,
    inputs=[
        gr.Image(label="Upload image"),
        gr.Textbox(placeholder="Enter text descriptions", label="Texts")
    ],
    outputs=gr.HTML(label="Segmentation Output")
    # outputs=gr.Image(label="Segmentation Output", height=450, width=450),
    # css="footer{display:none !important}"
)

iface.launch(debug=True, share=True)