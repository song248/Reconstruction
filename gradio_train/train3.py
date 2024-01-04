import gradio as gr

def greet(name, is_morning, temperature):
    if is_morning:
        text = 'good morning'
    else:
        text = 'good night'
    
    text = f'{name} sir, '+text
    celcius = (temperature-32)*5/9
    
    return text, celcius
    
demo = gr.Interface(
    fn =greet,
    inputs=['text', 'checkbox', gr.Slider(0, 100)],
    outputs=['text', 'number']
)

demo.launch(debug=True, share=True)