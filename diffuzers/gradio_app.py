import os
from dataclasses import dataclass
from typing import Optional

import gradio as gr
import torch
from PIL.PngImagePlugin import PngInfo

from .img2img import Img2Img
from .text2img import Text2Image


@dataclass
class Diffuzers:
    model: str
    output_path: str
    img2img_model: Optional[str] = None
    inpainting_model: Optional[str] = None

    def __post_init__(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.text2img = Text2Image(
            model=self.model,
            device=device,
        )
        self.img2img = Img2Img(
            model=self.img2img_model,
            device=device,
            text2img_model=self.text2img.pipeline,
        )
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "text2img"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "img2img"), exist_ok=True)

    def _text2img_input(self):
        with gr.Column():
            project_name_text = "Project name (optional; used to save the images, if provided)"
            project_name = gr.Textbox(label=project_name_text, lines=1, max_lines=1, placeholder="my_project")
            # TODO: add batch support
            # with gr.Tabs():
            #     with gr.TabItem("single"):
            prompt = gr.Textbox(label="Prompt", lines=3, max_lines=3)
            # with gr.TabItem("batch"):
            #     prompt = gr.File(file_types=["text"])
            # with gr.Tabs():
            #     with gr.TabItem("single"):
            negative_prompt = gr.Textbox(label="Negative prompt (optional)", lines=3, max_lines=3)
            # with gr.TabItem("batch"):
            #     negative_prompt = gr.File(file_types=["text"])
        with gr.Column():
            available_schedulers = list(self.text2img.compatible_schedulers.keys())
            scheduler = gr.Dropdown(choices=available_schedulers, label="Scheduler", value=available_schedulers[0])
            image_size = gr.Number(512, label="Image size", precision=0)
            guidance_scale = gr.Slider(1, maximum=20, value=7.5, step=0.5, label="Guidance scale")
            num_images = gr.Slider(1, 128, 1, label="Number of images per prompt", step=4)
            steps = gr.Slider(1, 150, 50, label="Steps")
            seed = gr.Slider(minimum=1, step=1, maximum=999999, randomize=True, label="Seed")
            generate_button = gr.Button("Generate")
        params_dict = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": scheduler,
            "image_size": image_size,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "steps": steps,
            "seed": seed,
            "project_name": project_name,
            "generate_button": generate_button,
        }
        return params_dict

    def _text2img_output(self):
        with gr.Column():
            text2img_output = gr.Gallery()
            text2img_output.style(grid=[4], container=False)
        with gr.Column():
            text2img_output_params = gr.Markdown()
        params_dict = {
            "output": text2img_output,
            "markdown": text2img_output_params,
        }
        return params_dict

    def _text2img_generate(
        self, prompt, negative_prompt, scheduler, image_size, guidance_scale, num_images, steps, seed, project_name
    ):
        output_images = self.text2img.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            scheduler=scheduler,
            image_size=image_size,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
            num_images=num_images,
        )
        params_used = f" - **Prompt:** {prompt}"
        params_used += f"\n - **Negative prompt:** {negative_prompt}"
        params_used += f"\n - **Scheduler:** {scheduler}"
        params_used += f"\n - **Image size:** {image_size}"
        params_used += f"\n - **Guidance scale:** {guidance_scale}"
        params_used += f"\n - **Steps:** {steps}"
        params_used += f"\n - **Seed:** {seed}"

        if len(project_name.strip()) > 0:
            self._save_images(
                images=output_images,
                metadata=params_used,
                folder_name=project_name,
                prefix="text2img",
            )

        return [output_images, params_used]

    def _img2img_input(self):
        with gr.Column():
            input_image = gr.Image(source="upload", label="input image | size must match model", type="pil")
            strength = gr.Slider(0, 1, 0.8, label="Strength")
            available_schedulers = list(self.img2img.compatible_schedulers.keys())
            scheduler = gr.Dropdown(choices=available_schedulers, label="Scheduler", value=available_schedulers[0])
            image_size = gr.Number(512, label="Image size (image will be resized to this)", precision=0)
            guidance_scale = gr.Slider(1, maximum=20, value=7.5, step=0.5, label="Guidance scale")
            num_images = gr.Slider(4, 128, 4, label="Number of images", step=4)
            steps = gr.Slider(1, 150, 50, label="Steps")
        with gr.Column():
            project_name_text = "Project name (optional; used to save the images, if provided)"
            project_name = gr.Textbox(label=project_name_text, lines=1, max_lines=1, placeholder="my_project")
            prompt = gr.Textbox(label="Prompt", lines=3, max_lines=3)
            negative_prompt = gr.Textbox(label="Negative prompt (optional)", lines=3, max_lines=3)
            seed = gr.Slider(minimum=1, step=1, maximum=999999, randomize=True, label="Seed")
            generate_button = gr.Button("Generate")
        params_dict = {
            "input_image": input_image,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "strength": strength,
            "scheduler": scheduler,
            "image_size": image_size,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "steps": steps,
            "seed": seed,
            "project_name": project_name,
            "generate_button": generate_button,
        }
        return params_dict

    def _img2img_output(self):
        with gr.Column():
            img2img_output = gr.Gallery()
            img2img_output.style(grid=[4], container=False)
        with gr.Column():
            img2img_output_params = gr.Markdown()
        params_dict = {
            "output": img2img_output,
            "markdown": img2img_output_params,
        }
        return params_dict

    def _img2img_generate(
        self,
        input_image,
        prompt,
        negative_prompt,
        strength,
        scheduler,
        image_size,
        guidance_scale,
        num_images,
        steps,
        seed,
        project_name,
    ):
        output_images = self.img2img.generate_image(
            image=input_image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            strength=strength,
            scheduler=scheduler,
            image_size=image_size,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
            num_images=num_images,
        )
        params_used = f" - **Prompt:** {prompt}"
        params_used += f"\n - **Negative prompt:** {negative_prompt}"
        params_used += f"\n - **Scheduler:** {scheduler}"
        params_used += f"\n - **Strength:** {strength}"
        params_used += f"\n - **Image size:** {image_size}"
        params_used += f"\n - **Guidance scale:** {guidance_scale}"
        params_used += f"\n - **Steps:** {steps}"
        params_used += f"\n - **Seed:** {seed}"

        if len(project_name.strip()) > 0:
            self._save_images(
                images=output_images,
                metadata=params_used,
                folder_name=project_name,
                prefix="img2img",
            )

        return [output_images, params_used]

    def _save_images(self, images, metadata, folder_name, prefix):
        folder_path = os.path.join(self.output_path, prefix, folder_name)
        os.makedirs(folder_path, exist_ok=True)
        for idx, image in enumerate(images):
            text2img_metadata = PngInfo()
            text2img_metadata.add_text(prefix, metadata)
            image.save(os.path.join(folder_path, f"{idx}.png"), format="PNG", pnginfo=text2img_metadata)
        with open(os.path.join(folder_path, "metadata.txt"), "w") as f:
            f.write(metadata)

    def _png_info(self, img):
        text2img_md = img.info.get("text2img", "")
        img2img_md = img.info.get("img2img", "")
        return_text = ""
        if len(text2img_md) > 0:
            return_text += f"## Text2Img\n{text2img_md}\n"
        if len(img2img_md) > 0:
            return_text += f"## Img2Img\n{img2img_md}\n"
        return return_text

    def app(self):
        with gr.Blocks() as demo:
            with gr.Blocks():
                gr.Markdown("# Diffuzers")
                gr.Markdown(f"Text2Img Model: {self.model}")
                if self.img2img_model:
                    gr.Markdown(f"Img2Img Model: {self.img2img_model}")
                else:
                    gr.Markdown(f"Img2Img Model: {self.model}")

            with gr.Tabs():
                with gr.TabItem("Text2Image", id="text2image"):
                    with gr.Row():
                        text2img_input = self._text2img_input()
                    with gr.Row():
                        text2img_output = self._text2img_output()
                        text2img_input["generate_button"].click(
                            fn=self._text2img_generate,
                            inputs=[
                                text2img_input["prompt"],
                                text2img_input["negative_prompt"],
                                text2img_input["scheduler"],
                                text2img_input["image_size"],
                                text2img_input["guidance_scale"],
                                text2img_input["num_images"],
                                text2img_input["steps"],
                                text2img_input["seed"],
                                text2img_input["project_name"],
                            ],
                            outputs=[text2img_output["output"], text2img_output["markdown"]],
                        )
                with gr.TabItem("Image2Image", id="img2img"):
                    with gr.Row():
                        img2img_input = self._img2img_input()
                    with gr.Row():
                        img2img_output = self._img2img_output()
                        img2img_input["generate_button"].click(
                            fn=self._img2img_generate,
                            inputs=[
                                img2img_input["input_image"],
                                img2img_input["prompt"],
                                img2img_input["negative_prompt"],
                                img2img_input["strength"],
                                img2img_input["scheduler"],
                                img2img_input["image_size"],
                                img2img_input["guidance_scale"],
                                img2img_input["num_images"],
                                img2img_input["steps"],
                                img2img_input["seed"],
                                img2img_input["project_name"],
                            ],
                            outputs=[img2img_output["output"], img2img_output["markdown"]],
                        )
                with gr.TabItem("Inpainting", id="inpainting"):
                    gr.Markdown("# coming soon!")
                with gr.TabItem("ImageInfo", id="imginfo"):
                    with gr.Column():
                        img_info_input_file = gr.Image(label="Input image", source="upload", type="pil")
                    with gr.Column():
                        img_info_output_md = gr.Markdown()
                    img_info_input_file.change(
                        fn=self._png_info,
                        inputs=[img_info_input_file],
                        outputs=[img_info_output_md],
                    )

        return demo
