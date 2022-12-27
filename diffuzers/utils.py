import os
import tempfile
import zipfile

import streamlit as st
import streamlit_ext as ste


def no_safety_checker(images, **kwargs):
    return images, False


def display_and_download_images(output_images, metadata):
    st.image(output_images, width=128, output_format="PNG")

    with st.spinner("Preparing images for download..."):
        # save images to a temporary directory
        with tempfile.TemporaryDirectory() as tmpdir:
            for i, image in enumerate(output_images):
                image.save(os.path.join(tmpdir, f"{i + 1}.png"), pnginfo=metadata)

            # zip the images
            zip_path = os.path.join(tmpdir, "images.zip")
            with zipfile.ZipFile(zip_path, "w") as zip:
                for filename in os.listdir(tmpdir):
                    if filename.endswith(".png"):
                        zip.write(os.path.join(tmpdir, filename), filename)

            # download the zip
            ste.download_button(
                label="Download images",
                data=open(zip_path, "rb").read(),
                file_name="images.zip",
                mime="application/zip",
            )
