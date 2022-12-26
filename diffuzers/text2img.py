from dataclasses import dataclass
from typing import Optional

import torch
from diffusers import DiffusionPipeline
from loguru import logger

from . import utils


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
        return output_images
