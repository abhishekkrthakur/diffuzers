# diffuzers

A web ui for [🤗 diffusers](https://github.com/huggingface/diffusers).

< under development, request features using issues, prs not accepted atm >

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
    diffuzers run --model_path <path to model> --image_size <image size>
    ``` 
    
For example, to run the web app for the [stable diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) model, run the following command:


    ```bash
    diffuzers run --model_path stabilityai/stable-diffusion-2-1 --image_size 768
    ```

Please note to use the correct image size for the model. For example, for the [stable diffusion 2.1](https://huggingface.co/stabilityai/stable-diffusion-2-1) model, the image size is 768. For the [stable diffusion 2.1 base](https://huggingface.co/stabilityai/stable-diffusion-2-1-base) model, the image size is 512.


The webapp runs on port 7860.