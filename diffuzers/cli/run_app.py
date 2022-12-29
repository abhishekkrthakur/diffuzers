import subprocess
from argparse import ArgumentParser

import torch

from . import BaseDiffuzersCommand


def run_app_command_factory(args):
    return RunDiffuzersAppCommand(
        args.model,
        args.inpainting_model,
        args.output_path,
        args.share,
        args.port,
        args.host,
        args.device,
    )


class RunDiffuzersAppCommand(BaseDiffuzersCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_app_parser = parser.add_parser(
            "run",
            description="✨ Run diffuzers app",
        )
        run_app_parser.add_argument(
            "--model",
            type=str,
            required=True,
            help="Path to model. This model will be used for text2img and img2img tasks.",
        )
        run_app_parser.add_argument(
            "--inpainting_model",
            type=str,
            required=False,
            help="Inpainting/Outpainting model. If not provided, the default model will be used which is sd2.0 inpainting model",
            default="stabilityai/stable-diffusion-2-inpainting",
        )
        run_app_parser.add_argument(
            "--output_path",
            type=str,
            required=False,
            help="Output path is optional, but if provided, all generations will automatically be saved to this path.",
        )
        run_app_parser.add_argument(
            "--share",
            action="store_true",
            help="Share the app",
        )
        run_app_parser.add_argument(
            "--port",
            type=int,
            default=10000,
            help="Port to run the app on",
            required=False,
        )
        run_app_parser.add_argument(
            "--host",
            type=str,
            default="127.0.0.1",
            help="Host to run the app on",
            required=False,
        )
        run_app_parser.add_argument(
            "--device",
            type=str,
            required=False,
            help="Device to use, e.g. cpu, cuda, cuda:0, mps (for m1 mac) etc.",
        )
        run_app_parser.set_defaults(func=run_app_command_factory)

    def __init__(self, model, inpainting_model, output_path, share, port, host, device):
        self.model = model
        self.inpainting_model = inpainting_model
        self.output_path = output_path
        self.share = share
        self.port = port
        self.host = host
        self.device = device

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        # from ..app import Diffuzers

        # print(self.share)
        # app = Diffuzers(self.model, self.output_path).app()
        # app.launch(show_api=False, share=self.share, server_port=self.port, server_name=self.host)
        import os

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "..", "app.py")
        cmd = [
            "streamlit",
            "run",
            filename,
            "--browser.gatherUsageStats",
            "false",
            "--browser.serverAddress",
            self.host,
            "--server.port",
            str(self.port),
            "--theme.base",
            "light",
            "--",
            "--model",
            self.model,
            "--inpainting_model",
            self.inpainting_model,
            "--device",
            self.device,
        ]
        if self.output_path is not None:
            cmd.extend(["--output_path", self.output_path])

        subprocess.run(cmd)
