"""Deploy Mediosis to Modal:  modal deploy modal_app.py"""
import modal

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0")
    .pip_install("torch", "torchvision", index_url="https://download.pytorch.org/whl/cpu")
    .pip_install("tensorflow-cpu>=2.16", "keras>=3", "torchxrayvision>=1.2", "gensim>=4.3.3",
                 "pytorch-crf>=0.7.2", "rapidfuzz>=3", "seqeval>=1.2.2", "easyocr>=1.7",
                 "gradio>=5", "fastapi[standard]", "pandas", "scikit-learn", "matplotlib", "pillow")
    .add_local_dir(".", "/root/app", copy=True,
                   ignore=[".git", ".github", ".venv", "artifacts", "**/__pycache__", "*.ipynb", "*.zip"])
    .run_commands(
        "cd /root/app && python make_samples.py && python train.py",
        "python -c \"import torchxrayvision as xrv; xrv.models.DenseNet(weights='densenet121-res224-all')\"",
        "python -c \"import easyocr; easyocr.Reader(['en'], gpu=False)\"",
    )
)

app = modal.App("mediosis", image=image)


@app.function(cpu=2.0, memory=4096, timeout=600, scaledown_window=300, max_containers=1)
@modal.concurrent(max_inputs=50)
@modal.asgi_app()
def web():
    import os
    import sys
    os.chdir("/root/app")
    sys.path.insert(0, "/root/app")
    import gradio as gr
    from fastapi import FastAPI
    import app as mediosis_app
    return gr.mount_gradio_app(FastAPI(), mediosis_app.demo, path="/")