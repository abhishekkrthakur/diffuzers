import base64
import os
import tempfile
import zipfile
from datetime import datetime

import streamlit as st
import streamlit_ext as ste
from loguru import logger
from PIL.PngImagePlugin import PngInfo
from st_clickable_images import clickable_images


def no_safety_checker(images, **kwargs):
    return images, False


def save_images(images, module, metadata, output_path):
    if output_path is None:
        logger.warning("No output path specified, skipping saving images")
        return
    current_datetime = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(f"{output_path}/{module}", exist_ok=True)
    os.makedirs(f"{output_path}/{module}/{current_datetime}", exist_ok=True)

    _metadata = PngInfo()
    _metadata.add_text("text2img", metadata)

    for i, img in enumerate(images):
        img.save(
            f"{output_path}/{module}/{current_datetime}/{i}.png",
            pnginfo=_metadata,
        )

    # save metadata as text file
    with open(f"{output_path}/{module}/{current_datetime}/metadata.txt", "w") as f:
        f.write(metadata)
    logger.info(f"Saved images to {output_path}/{module}/{current_datetime}")


def display_and_download_images(output_images, metadata):
    # st.image(output_images, width=128, output_format="PNG")

    with st.spinner("Preparing images for download..."):
        # save images to a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            gallery_images = []
            for i, image in enumerate(output_images):
                image.save(os.path.join(tmpdir, f"{i + 1}.png"), pnginfo=metadata)
                with open(os.path.join(tmpdir, f"{i + 1}.png"), "rb") as img:
                    encoded = base64.b64encode(img.read()).decode()
                    gallery_images.append(f"data:image/jpeg;base64,{encoded}")

            # zip the images
            zip_path = os.path.join(tmpdir, "images.zip")
            with zipfile.ZipFile(zip_path, "w") as zip:
                for filename in os.listdir(tmpdir):
                    if filename.endswith(".png"):
                        zip.write(os.path.join(tmpdir, filename), filename)

            # download the zip
            ste.download_button(
                label="Download",
                data=open(zip_path, "rb").read(),
                file_name="images.zip",
                mime="application/zip",
            )

            _ = clickable_images(
                gallery_images,
                titles=[f"Image #{str(i)}" for i in range(len(gallery_images))],
                div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
                img_style={"margin": "5px", "height": "200px"},
            )
