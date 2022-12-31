import gc
import json
from dataclasses import dataclass
from typing import Optional

import streamlit as st
import torch
from diffusers import StableDiffusionUpscalePipeline
from loguru import logger
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from diffuzers import utils


@dataclass
class Upscaler:
    model: str = "stabilityai/stable-diffusion-x4-upscaler"
    device: Optional[str] = None
    output_path: Optional[str] = None

    def __str__(self) -> str:
        return f"Upscaler(model={self.model}, device={self.device}, output_path={self.output_path})"

    def __post_init__(self):
        self.pipeline = StableDiffusionUpscalePipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker
        self._compatible_schedulers = self.pipeline.scheduler.compatibles
        self.scheduler_config = self.pipeline.scheduler.config
        self.compatible_schedulers = {scheduler.__name__: scheduler for scheduler in self._compatible_schedulers}

    def _set_scheduler(self, scheduler_name):
        scheduler = self.compatible_schedulers[scheduler_name].from_config(self.scheduler_config)
        self.pipeline.scheduler = scheduler

    def generate_image(
        self, image, prompt, negative_prompt, guidance_scale, noise_level, num_images, eta, scheduler, steps, seed
    ):
        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)
        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
            generator = torch.Generator(device=self.device).manual_seed(seed)
        num_images = int(num_images)
        output_images = self.pipeline(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            noise_level=noise_level,
            num_inference_steps=steps,
            eta=eta,
            num_images_per_prompt=num_images,
            generator=generator,
            guidance_scale=guidance_scale,
        ).images

        torch.cuda.empty_cache()
        gc.collect()
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "noise_level": noise_level,
            "num_images": num_images,
            "eta": eta,
            "scheduler": scheduler,
            "steps": steps,
            "seed": seed,
        }

        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("upscaler", metadata)

        utils.save_images(
            images=output_images,
            module="upscaler",
            metadata=metadata,
            output_path=self.output_path,
        )

        return output_images, _metadata

    def app(self):
        available_schedulers = list(self.compatible_schedulers.keys())
        # if EulerAncestralDiscreteScheduler is available in available_schedulers, move it to the first position
        if "EulerAncestralDiscreteScheduler" in available_schedulers:
            available_schedulers.insert(
                0, available_schedulers.pop(available_schedulers.index("EulerAncestralDiscreteScheduler"))
            )

        input_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])
        if input_image is not None:
            input_image = Image.open(input_image)
            input_image = input_image.convert("RGB").resize((128, 128), resample=Image.LANCZOS)
            st.image(input_image, use_column_width=True)

        col1, col2 = st.columns(2)
        with col1:
            prompt = st.text_area("Prompt (Optional)", "")
        with col2:
            negative_prompt = st.text_area("Negative Prompt (Optional)", "")

        scheduler = st.sidebar.selectbox("Scheduler", available_schedulers, index=0)
        guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 40.0, 7.5, 0.5)
        noise_level = st.sidebar.slider("Noise level", 0, 100, 20, 1)
        eta = st.sidebar.slider("Eta", 0.0, 1.0, 0.0, 0.1)
        num_images = st.sidebar.slider("Number of images per prompt", 1, 30, 1, 1)
        steps = st.sidebar.slider("Steps", 1, 150, 50, 1)
        seed_placeholder = st.sidebar.empty()
        seed = seed_placeholder.number_input("Seed", value=42, min_value=1, max_value=999999, step=1)
        random_seed = st.sidebar.button("Random seed")
        _seed = torch.randint(1, 999999, (1,)).item()
        if random_seed:
            seed = seed_placeholder.number_input("Seed", value=_seed, min_value=1, max_value=999999, step=1)
        sub_col, download_col = st.columns(2)
        with sub_col:
            submit = st.button("Generate")

        if submit:
            with st.spinner("Generating images..."):
                output_images, metadata = self.generate_image(
                    prompt=prompt,
                    image=input_image,
                    negative_prompt=negative_prompt,
                    scheduler=scheduler,
                    num_images=num_images,
                    guidance_scale=guidance_scale,
                    steps=steps,
                    seed=seed,
                    noise_level=noise_level,
                    eta=eta,
                )

            utils.display_and_download_images(output_images, metadata, download_col)
