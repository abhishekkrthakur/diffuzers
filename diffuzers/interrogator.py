import os
from dataclasses import dataclass
from typing import Optional

import streamlit as st
from clip_interrogator import Config, Interrogator
from huggingface_hub import hf_hub_download
from loguru import logger
from PIL import Image

from diffuzers import utils


@dataclass
class ImageInterrogator:
    device: Optional[str] = None
    output_path: Optional[str] = None

    def inference(self, model, image, mode):
        preprocess_files = [
            "ViT-H-14_laion2b_s32b_b79k_artists.pkl",
            "ViT-H-14_laion2b_s32b_b79k_flavors.pkl",
            "ViT-H-14_laion2b_s32b_b79k_mediums.pkl",
            "ViT-H-14_laion2b_s32b_b79k_movements.pkl",
            "ViT-H-14_laion2b_s32b_b79k_trendings.pkl",
            "ViT-L-14_openai_artists.pkl",
            "ViT-L-14_openai_flavors.pkl",
            "ViT-L-14_openai_mediums.pkl",
            "ViT-L-14_openai_movements.pkl",
            "ViT-L-14_openai_trendings.pkl",
        ]

        logger.info("Downloading preprocessed cache files...")
        for file in preprocess_files:
            path = hf_hub_download(repo_id="pharma/ci-preprocess", filename=file, cache_dir=utils.cache_folder())
            cache_path = os.path.dirname(path)

        config = Config(cache_path=cache_path, clip_model_path=utils.cache_folder(), clip_model_name=model)
        pipeline = Interrogator(config)

        pipeline.config.blip_num_beams = 64
        pipeline.config.chunk_size = 2048
        pipeline.config.flavor_intermediate_count = 2048 if model == "ViT-L-14/openai" else 1024

        if mode == "best":
            prompt = pipeline.interrogate(image)
        elif mode == "classic":
            prompt = pipeline.interrogate_classic(image)
        else:
            prompt = pipeline.interrogate_fast(image)
        return prompt

    def app(self):
        # upload image
        input_image = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        with st.form(key="image_interrogator"):
            clip_model = st.selectbox("CLIP Model", ["ViT-L (Best for SD1.X)", "ViT-H (Best for SD2.X)"])
            mode = st.selectbox("Mode", ["Best", "Classic"])
            submit = st.form_submit_button("Interrogate")
            if input_image is not None:
                # read image using pil
                pil_image = Image.open(input_image).convert("RGB")
            if submit:
                with st.spinner("Interrogating image..."):
                    if clip_model == "ViT-L (Best for SD1.X)":
                        model = "ViT-L-14/openai"
                    else:
                        model = "ViT-H-14/laion2b_s32b_b79k"
                    prompt = self.inference(model, pil_image, mode.lower())
                col1, col2 = st.columns(2)
                with col1:
                    st.image(input_image, use_column_width=True)
                with col2:
                    st.write(prompt)
