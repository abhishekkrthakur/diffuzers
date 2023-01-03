import subprocess
from argparse import ArgumentParser

import torch
from pyngrok import ngrok

from . import BaseDiffuzersCommand


def run_app_command_factory(args):
    return RunDiffuzersAppCommand(
        args.output,
        args.share,
        args.port,
        args.host,
        args.device,
        args.ngrok_key,
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
        run_app_parser.add_argument(
            "--ngrok_key",
            type=str,
            required=False,
            help="Ngrok key to use for sharing the app. Only required if you want to share the app",
        )
        run_app_parser.set_defaults(func=run_app_command_factory)

    def __init__(self, output, share, port, host, device, ngrok_key):
        self.output = output
        self.share = share
        self.port = port
        self.host = host
        self.device = device
        self.ngrok_key = ngrok_key

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self.share:
            if self.ngrok_key is None:
                raise ValueError(
                    "ngrok key is required if you want to share the app. Get it for free from https://dashboard.ngrok.com/get-started/your-authtoken"
                )

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

        if self.share:
            ngrok.set_auth_token(self.ngrok_key)
            public_url = ngrok.connect(self.port).public_url
            print(f"Sharing app at {public_url}")

        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            shell=False,
            universal_newlines=True,
            bufsize=1,
        )
        with proc as p:
            try:
                for line in p.stdout:
                    print(line, end="")
            except KeyboardInterrupt:
                print("Killing streamlit app")
                p.kill()
                if self.share:
                    print("Killing ngrok tunnel")
                    ngrok.kill()
                p.wait()
                raise
