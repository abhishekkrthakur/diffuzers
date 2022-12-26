import os
from dataclasses import dataclass

import gradio as gr

from .text2img import Text2Image


@dataclass
class Diffuzers:
    model_path: str
    image_size: int
    output_path: str

    def __post_init__(self):
        self.text2img = Text2Image(
            model=self.model_path,
            device="cuda",
            image_size=self.image_size,
        )
        os.makedirs(self.output_path, exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "text2img"), exist_ok=True)

    def _text2img_input(self):
        prompt = gr.Textbox(label="Prompt", lines=3, max_lines=3)
        negative_prompt = gr.Textbox(label="Negative prompt (optional)", lines=3, max_lines=3)
        available_schedulers = list(self.text2img.compatible_schedulers.keys())
        scheduler = gr.Dropdown(choices=available_schedulers, label="Scheduler", value=available_schedulers[0])
        guidance_scale = gr.Slider(1, maximum=20, value=7.5, step=0.5, label="Guidance scale")
        num_images = gr.Slider(4, 128, 4, label="Number of images", step=4)
        steps = gr.Slider(1, 150, 50, label="Steps")
        seed = gr.Slider(minimum=1, step=1, maximum=999999, randomize=True, label="Seed")
        generate_button = gr.Button("Generate")
        params_dict = {
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "scheduler": scheduler,
            "guidance_scale": guidance_scale,
            "num_images": num_images,
            "steps": steps,
            "seed": seed,
            "generate_button": generate_button,
        }
        return params_dict

    def _text2img_output(self):
        text2img_output = gr.Gallery()
        text2img_output.style(grid=[4], container=False)
        text2img_output_params = gr.Markdown()
        params_dict = {
            "output": text2img_output,
            "markdown": text2img_output_params,
        }
        return params_dict

    def _text2img_generate(self, prompt, negative_prompt, scheduler, guidance_scale, num_images, steps, seed):
        output_images = self.text2img.generate_image(
            prompt=prompt,
            negative_prompt=negative_prompt,
            scheduler=scheduler,
            guidance_scale=guidance_scale,
            steps=steps,
            seed=seed,
            num_images=num_images,
        )
        params_used = f"**Prompt:** {prompt} | **Negative prompt:** {negative_prompt} | **Scheduler:** {scheduler}"
        params_used += f" | **Guidance scale:** {guidance_scale} | **Steps:** {steps} | **Seed:** {seed}"
        return [
            output_images,
            params_used,
            gr.Text.update(visible=True, interactive=True),
            gr.Button.update(visible=True),
        ]

    def _save_images(self, images, metadata, folder_name):
        folder_path = os.path.join(self.output_path, "text2img", folder_name)
        os.makedirs(folder_path, exist_ok=True)
        for image in images:
            image = image["name"]
            base_name = os.path.basename(image)
            os.system(f"cp {image} {folder_path}/{base_name}")
        with open(os.path.join(folder_path, "metadata.txt"), "w") as f:
            f.write(metadata)
        return [gr.Text.update(visible=False), gr.Button.update(visible=False)]

    def app(self):
        with gr.Blocks() as demo:
            with gr.Blocks():
                gr.Markdown("# Diffuzers")
            with gr.Tabs():
                with gr.TabItem("Text2Image", id="text2image"):
                    with gr.Row():
                        with gr.Column():
                            text2img_input = self._text2img_input()
                        with gr.Column():
                            text2img_output = self._text2img_output()
                            text2img_folder_name = gr.Text(lines=1, label="Folder name", max_lines=1, visible=False)
                            text2img_save_button = gr.Button("Save", visible=False)
                            text2img_input["generate_button"].click(
                                fn=self._text2img_generate,
                                inputs=[
                                    text2img_input["prompt"],
                                    text2img_input["negative_prompt"],
                                    text2img_input["scheduler"],
                                    text2img_input["guidance_scale"],
                                    text2img_input["num_images"],
                                    text2img_input["steps"],
                                    text2img_input["seed"],
                                ],
                                outputs=[
                                    text2img_output["output"],
                                    text2img_output["markdown"],
                                    text2img_folder_name,
                                    text2img_save_button,
                                ],
                            )
                            text2img_save_button.click(
                                fn=self._save_images,
                                inputs=[text2img_output["output"], text2img_output["markdown"], text2img_folder_name],
                                outputs=[text2img_folder_name, text2img_save_button],
                            )
                with gr.TabItem("Image2Image", id="img2img"):
                    gr.Markdown("# coming soon!")
                with gr.TabItem("Inpainting", id="inpainting"):
                    gr.Markdown("# coming soon!")

        return demo
