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
    model: str
    device: Optional[str] = None
    output_path: Optional[str] = None

    def __post_init__(self):
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

        config = Config(cache_path=cache_path, clip_model_path=utils.cache_folder(), clip_model_name=self.model)
        self.pipeline = Interrogator(config)

        self.pipeline.config.blip_num_beams = 64
        self.pipeline.config.chunk_size = 2048
        self.pipeline.config.flavor_intermediate_count = 2048 if self.model == "ViT-L-14/openai" else 1024

    def app(self):
        # upload image
        uploaded_file = st.file_uploader("Choose an image...", type=["png", "jpg", "jpeg"])
        if uploaded_file is not None:
            # read image using pil
            pil_image = Image.open(uploaded_file)
            st.image(uploaded_file, use_column_width=True)
