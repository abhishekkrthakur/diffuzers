import argparse

import streamlit as st
import torch
from loguru import logger

from diffuzers.image_info import ImageInfo
from diffuzers.img2img import Img2Img
from diffuzers.inpainting import Inpainting
from diffuzers.text2img import Text2Image


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Path to model",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Output path",
    )
    parser.add_argument(
        "--inpainting_model",
        type=str,
        required=True,
        help="Path to inpainting model",
    )
    return parser.parse_args()


@st.experimental_singleton
def get_models(_args, device):
    text2img = Text2Image(
        model=_args.model,
        device=device,
        output_path=_args.output_path,
    )
    inpainting = Inpainting(
        model=args.inpainting_model,
        device=device,
        output_path=_args.output_path,
    )
    img2img = Img2Img(
        model=_args.model,
        device=device,
        text2img_model=text2img.pipeline,
        output_path=_args.output_path,
    )
    return text2img, img2img, inpainting


def run_app(args, text2img, img2img, inpainting):
    st.sidebar.title("Diffuzers")
    task = st.selectbox(
        "Task",
        [
            "Text2Img",
            "Img2Img",
            # "Outpainting",
            "Inpainting",
            "ImageInfo",
        ],
    )
    if task == "Text2Img":
        text2img.app()
    elif task == "Img2Img":
        img2img.app()
    # elif task == "Outpainting":
    #     st.write("Outpainting")
    elif task == "Inpainting":
        inpainting.app()
    elif task == "ImageInfo":
        ImageInfo().app()


if __name__ == "__main__":
    args = parse_args()
    logger.info(f"Args: {args}")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    text2img, img2img, inpainting = get_models(args, device)
    run_app(args, text2img, img2img, inpainting)
