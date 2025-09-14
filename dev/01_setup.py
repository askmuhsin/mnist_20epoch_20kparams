import marimo

__generated_with = "0.15.3"
app = marimo.App(width="full")


@app.cell
def _():
    import os
    import torch
    import lightning as L
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    return L, MNIST, ToTensor, os, torch


@app.cell
def _(torch):
    torch.__version__
    return


@app.cell
def _(torch):
    torch.mps.is_available()
    return


@app.cell
def _(torch):
    torch.mps.device_count()
    f'{torch.mps.recommended_max_memory() / (1024**3):.2f} GB'
    return


@app.cell
def _(L):
    L.__version__
    return


@app.cell
def _():
    return


@app.cell
def _(MNIST, ToTensor, os):
    data_path = os.path.join(os.getcwd(), 'data')
    dataset = MNIST(data_path, download=True, transform=ToTensor())
    return (dataset,)


@app.cell
def _(dataset):
    dataset
    return


@app.cell
def _():
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
