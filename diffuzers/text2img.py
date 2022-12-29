import gc
import json
from dataclasses import dataclass
from typing import Optional

import streamlit as st
import torch
from diffusers import DiffusionPipeline
from loguru import logger
from PIL.PngImagePlugin import PngInfo

from diffuzers import utils


@dataclass
class Text2Image:
    device: Optional[str] = None
    model: Optional[str] = None
    output_path: Optional[str] = None

    def __post_init__(self):
        self.pipeline = DiffusionPipeline.from_pretrained(
            self.model,
            torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
        )
        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker
        self._compatible_schedulers = self.pipeline.scheduler.compatibles
        self.scheduler_config = self.pipeline.scheduler.config
        self.compatible_schedulers = {scheduler.__name__: scheduler for scheduler in self._compatible_schedulers}

        if self.device == "mps":
            self.pipeline.enable_attention_slicing()
            # warmup
            prompt = "a photo of an astronaut riding a horse on mars"
            _ = self.pipeline(prompt, num_inference_steps=1)

    def _set_scheduler(self, scheduler_name):
        scheduler = self.compatible_schedulers[scheduler_name].from_config(self.scheduler_config)
        self.pipeline.scheduler = scheduler

    def generate_image(self, prompt, negative_prompt, scheduler, image_size, num_images, guidance_scale, steps, seed):
        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        num_images = int(num_images)
        output_images = self.pipeline(
            prompt,
            negative_prompt=negative_prompt,
            width=image_size,
            height=image_size,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        ).images
        torch.cuda.empty_cache()
        gc.collect()
        metadata = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": scheduler,
            "image_size": image_size,
            "num_images": num_images,
            "guidance_scale": guidance_scale,
            "steps": steps,
            "seed": seed,
        }
        metadata = json.dumps(metadata)
        _metadata = PngInfo()
        _metadata.add_text("text2img", metadata)

        utils.save_images(
            images=output_images,
            module="text2img",
            metadata=metadata,
            output_path=self.output_path,
        )

        return output_images, _metadata

    def app(self):
        available_schedulers = list(self.compatible_schedulers.keys())
        if "EulerAncestralDiscreteScheduler" in available_schedulers:
            available_schedulers.insert(
                0, available_schedulers.pop(available_schedulers.index("EulerAncestralDiscreteScheduler"))
            )
        with st.form(key="text2img"):
            col1, col2 = st.columns(2)
            with col1:
                prompt = st.text_area("Prompt", "Blue elephant")
            with col2:
                negative_prompt = st.text_area("Negative Prompt", "")
            # sidebar options
            scheduler = st.sidebar.selectbox("Scheduler", available_schedulers, index=0)
            image_size = st.sidebar.slider("Image size", 256, 1024, 512, 256)
            guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 40.0, 7.5, 0.5)
            num_images = st.sidebar.slider("Number of images per prompt", 1, 30, 1, 1)
            steps = st.sidebar.slider("Steps", 1, 150, 50, 1)
            seed = st.sidebar.slider("Seed", 1, 999999, 1, 1)
            submit = st.form_submit_button("Generate")

        if submit:
            with st.spinner("Generating images..."):
                output_images, metadata = self.generate_image(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    scheduler=scheduler,
                    image_size=image_size,
                    num_images=num_images,
                    guidance_scale=guidance_scale,
                    steps=steps,
                    seed=seed,
                )

            utils.display_and_download_images(output_images, metadata)
