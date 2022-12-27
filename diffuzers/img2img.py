import gc
from dataclasses import dataclass
from typing import Optional, Union

import torch
from diffusers import (
    AltDiffusionImg2ImgPipeline,
    AltDiffusionPipeline,
    DiffusionPipeline,
    StableDiffusionImg2ImgPipeline,
    StableDiffusionPipeline,
)
from loguru import logger

from . import utils


@dataclass
class Img2Img:
    model: Optional[str] = None
    device: Optional[str] = None
    text2img_model: Optional[Union[StableDiffusionPipeline, AltDiffusionPipeline]] = None

    def __post_init__(self):
        if self.model is not None:
            self.text2img_model = DiffusionPipeline.from_pretrained(
                self.model,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
            )
        components = self.text2img_model.components

        if isinstance(self.text2img_model, StableDiffusionPipeline):
            self.pipeline = StableDiffusionImg2ImgPipeline(**components)
        elif isinstance(self.text2img_model, AltDiffusionPipeline):
            self.pipeline = AltDiffusionImg2ImgPipeline(**components)
        else:
            raise ValueError("Model type not supported")

        self.pipeline.to(self.device)
        self.pipeline.safety_checker = utils.no_safety_checker
        self._compatible_schedulers = self.pipeline.scheduler.compatibles
        self.scheduler_config = self.pipeline.scheduler.config
        self.compatible_schedulers = {scheduler.__name__: scheduler for scheduler in self._compatible_schedulers}

    def _set_scheduler(self, scheduler_name):
        scheduler = self.compatible_schedulers[scheduler_name].from_config(self.scheduler_config)
        self.pipeline.scheduler = scheduler

    def generate_image(
        self, prompt, image, strength, negative_prompt, scheduler, image_size, num_images, guidance_scale, steps, seed
    ):
        self._set_scheduler(scheduler)
        logger.info(self.pipeline.scheduler)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        num_images = int(num_images)
        output_images = self.pipeline(
            prompt=prompt,
            image=image,
            strength=strength,
            negative_prompt=negative_prompt,
            num_inference_steps=steps,
            guidance_scale=guidance_scale,
            num_images_per_prompt=num_images,
            generator=generator,
        ).images
        torch.cuda.empty_cache()
        gc.collect()
        return output_images
