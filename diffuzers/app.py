from dataclasses import dataclass

import gradio as gr

from .text2img import Text2Image


@dataclass
class Diffuzers:
    model_path: str
    image_size: int

    def __post_init__(self):
        self.text2img = Text2Image(
            model=self.model_path,
            device="cuda",
            image_size=self.image_size,
        )

    def _text2img_input(self):
        prompt = gr.Textbox(label="Prompt", lines=3, max_lines=3)
        negative_prompt = gr.Textbox(label="Negative prompt (optional)", lines=3, max_lines=3)
        scheduler = gr.Dropdown(choices=list(self.text2img.compatible_schedulers.keys()), label="Scheduler")
        guidance_scale = gr.Slider(1, maximum=20, value=7.5, step=0.5, label="Guidance scale")
        # num_images = gr.Slider(1, 10, 1, label="Number of images", step=1)
        steps = gr.Slider(1, 150, 50, label="Steps")
        seed = gr.Slider(minimum=1, step=1, maximum=999999999999999999, randomize=True, label="Seed")
        generate_button = gr.Button("Generate")
        return prompt, negative_prompt, scheduler, guidance_scale, steps, seed, generate_button

    def _text2img_output(self):
        text2img_output = gr.Gallery()
        text2img_output.style(grid=[4], container=False)
        text2img_output_params = gr.Markdown()
        return text2img_output, text2img_output_params

    def _text2img_generate(self, prompt, negative_prompt, scheduler, guidance_scale, steps, seed):
        output_images = self.text2img.generate_image(prompt, negative_prompt, scheduler, guidance_scale, steps, seed)
        params = f"""
        - Prompt: {prompt}
        - Negative prompt: {negative_prompt}
        - Scheduler: {scheduler}
        - Guidance scale: {guidance_scale}
        - Steps: {steps}
        - Seed: {seed}
        """
        return output_images, params

    def app(self):
        with gr.Blocks() as demo:
            with gr.Tabs():
                with gr.TabItem("Text2Image", id="text2image"):
                    with gr.Row():
                        with gr.Column():
                            (
                                prompt,
                                negative_prompt,
                                scheduler,
                                guidance_scale,
                                steps,
                                seed,
                                generate_button,
                            ) = self._text2img_input()

                        with gr.Column():
                            text2img_output, text2img_output_params = self._text2img_output()
                            generate_button.click(
                                fn=self._text2img_generate,
                                inputs=[prompt, negative_prompt, scheduler, guidance_scale, steps, seed],
                                outputs=[text2img_output, text2img_output_params],
                            )
                with gr.TabItem("Image2Image", id="img2img"):
                    gr.Markdown("# Image2Image")

        return demo
