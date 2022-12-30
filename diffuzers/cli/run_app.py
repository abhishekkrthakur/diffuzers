import subprocess
from argparse import ArgumentParser

import torch

from . import BaseDiffuzersCommand


def run_app_command_factory(args):
    return RunDiffuzersAppCommand(
        args.output,
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
            "--output",
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

    def __init__(self, output, share, port, host, device):
        self.output = output
        self.share = share
        self.port = port
        self.host = host
        self.device = device

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def run(self):
        # from ..app import Diffuzers

        # print(self.share)
        # app = Diffuzers(self.model, self.output).app()
        # app.launch(show_api=False, share=self.share, server_port=self.port, server_name=self.host)
        import os

        dirname = os.path.dirname(__file__)
        filename = os.path.join(dirname, "..", "Home.py")
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
            "--device",
            self.device,
        ]
        if self.output is not None:
            cmd.extend(["--output", self.output])

        subprocess.run(cmd)
