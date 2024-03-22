import gradio as gr
from grok.run import main



demo = gr.Interface(
    fn=main, inputs="textbox", outputs="textbox")
    
demo.launch(share=True)
