# diffuzers

A web ui for [🤗 diffusers](https://github.com/huggingface/diffusers).

< under development, request features using issues, prs not accepted atm >

<a target="_blank" href="https://colab.research.google.com/github/abhishekkrthakur/diffuzers/blob/main/diffuzers.ipynb">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

![image](https://github.com/abhishekkrthakur/diffuzers/raw/main/static/screenshot.png)

## Features:

- text to image
- image to image
- textual inversion
- inpainting
- outpainting (coming soon)
- image info
- stable diffusion upscaler
- gfpgan
- clip interrogator
- need more? create an [issue](https://github.com/abhishekkrthakur/diffuzers/issues)


## Installation

To install bleeding edge version of diffuzers, clone the repo and install it using pip.

```bash
git clone https://github.com/abhishekkrthakur/diffuzers
cd diffuzers
pip install -e .
```

Installation using pip:
    
```bash 
pip install diffuzers
```

## Usage

To run the web app, run the following command:

```bash
diffuzers run
```

## All CLI Options:

```bash
❯ diffuzers run --help

usage: diffuzers <command> [<args>] run [-h] [--output OUTPUT] [--share] [--port PORT] [--host HOST]
                                        [--device DEVICE]

✨ Run diffuzers app

optional arguments:
  -h, --help       show this help message and exit
  --output OUTPUT  Output path is optional, but if provided, all generations will automatically be saved to this
                   path.
  --share          Share the app
  --port PORT      Port to run the app on
  --host HOST      Host to run the app on
  --device DEVICE  Device to use, e.g. cpu, cuda, cuda:0, mps (for m1 mac) etc.
```
