import gradio as gr
from bloom import run


demo = gr.Interface(
    fn=run, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)
