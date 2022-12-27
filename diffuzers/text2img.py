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

        return output_images, _metadata

    # def _text2img_input(self):
    #     with gr.Column():
    #         project_name_text = "Project name (optional; used to save the images, if provided)"
    #         project_name = gr.Textbox(label=project_name_text, lines=1, max_lines=1, placeholder="my_project")
    #         # TODO: add batch support
    #         # with gr.Tabs():
    #         #     with gr.TabItem("single"):
    #         prompt = gr.Textbox(label="Prompt", lines=3, max_lines=3)
    #         # with gr.TabItem("batch"):
    #         #     prompt = gr.File(file_types=["text"])
    #         # with gr.Tabs():
    #         #     with gr.TabItem("single"):
    #         negative_prompt = gr.Textbox(label="Negative prompt (optional)", lines=3, max_lines=3)
    #         # with gr.TabItem("batch"):
    #         #     negative_prompt = gr.File(file_types=["text"])
    #     with gr.Column():
    #         available_schedulers = list(self.text2img.compatible_schedulers.keys())
    #         scheduler = gr.Dropdown(choices=available_schedulers, label="Scheduler", value=available_schedulers[0])
    #         image_size = gr.Number(512, label="Image size", precision=0)
    #         guidance_scale = gr.Slider(1, maximum=20, value=7.5, step=0.5, label="Guidance scale")
    #         num_images = gr.Slider(1, 128, 1, label="Number of images per prompt", step=4)
    #         steps = gr.Slider(1, 150, 50, label="Steps")
    #         seed = gr.Slider(minimum=1, step=1, maximum=999999, randomize=True, label="Seed")
    #         generate_button = gr.Button("Generate")

    def app(self):
        col1, col2 = st.columns(2)
        with col1:
            prompt = st.text_area("Prompt", "Blue elephant")
        with col2:
            negative_prompt = st.text_area("Negative Prompt", "")
        submit = st.button("Generate")

        # sidebar options
        available_schedulers = list(self.compatible_schedulers.keys())
        scheduler = st.sidebar.selectbox("Scheduler", available_schedulers, index=0)
        image_size = st.sidebar.slider("Image size", 256, 1024, 512, 256)
        guidance_scale = st.sidebar.slider("Guidance scale", 1.0, 40.0, 7.5, 0.5)
        num_images = st.sidebar.slider("Number of images per prompt", 1, 30, 1, 1)
        steps = st.sidebar.slider("Steps", 1, 150, 50, 1)
        seed = st.sidebar.slider("Seed", 1, 999999, 1, 1)

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
