import numpy as np
import gradio as gr


def greet(name):
    return 'Hi'+name

# fn: 실행함수
# input: 입력형태
# output: 출력형태
demo = gr.Interface(fn=greet, inputs='text', outputs='text')

demo.launch(debug=True, share=True)