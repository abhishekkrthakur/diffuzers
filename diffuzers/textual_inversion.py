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


def load_embed(learned_embeds_path, text_encoder, tokenizer, token=None):
    loaded_learned_embeds = torch.load(learned_embeds_path, map_location="cpu")
    if len(loaded_learned_embeds) > 2:
        embeds = loaded_learned_embeds["string_to_param"]["*"][-1, :]
    else:
        # separate token and the embeds
        trained_token = list(loaded_learned_embeds.keys())[0]
        embeds = loaded_learned_embeds[trained_token]

    # add the token in tokenizer
    token = token if token is not None else trained_token
    num_added_tokens = tokenizer.add_tokens(token)
    i = 1
    while num_added_tokens == 0:
        logger.warning(f"The tokenizer already contains the token {token}.")
        token = f"{token[:-1]}-{i}>"
        logger.info(f"Attempting to add the token {token}.")
        num_added_tokens = tokenizer.add_tokens(token)
        i += 1

    # resize the token embeddings
    text_encoder.resize_token_embeddings(len(tokenizer))

    # get the id for the token and assign the embeds
    token_id = tokenizer.convert_tokens_to_ids(token)
    text_encoder.get_input_embeddings().weight.data[token_id] = embeds
    return token


@dataclass
class TextualInversion:
    model: str
    embeddings_url: str
    token_identifier: str
    device: Optional[str] = None
    output_path: Optional[str] = None

    def __str__(self) -> str:
        return f"TextualInversion(model={self.model}, embeddings={self.embeddings_url}, token_identifier={self.token_identifier}, device={self.device}, output_path={self.output_path})"

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

        # download the embeddings
        self.embeddings_path = utils.download_file(self.embeddings_url)
        load_embed(
            learned_embeds_path=self.embeddings_path,
            text_encoder=self.pipeline.text_encoder,
            tokenizer=self.pipeline.tokenizer,
            token=self.token_identifier,
        )

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
        if self.device == "mps":
            generator = torch.manual_seed(seed)
            num_images = 1
        else:
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
        _metadata.add_text("textual_inversion", metadata)

        utils.save_images(
            images=output_images,
            module="textual_inversion",
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
        sub_col, download_col = st.columns(2)
        with sub_col:
            submit = st.button("Generate")

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

            utils.display_and_download_images(output_images, metadata, download_col)
