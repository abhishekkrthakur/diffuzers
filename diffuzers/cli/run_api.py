import subprocess
from argparse import ArgumentParser

import torch

from . import BaseDiffuzersCommand


def run_api_command_factory(args):
    return RunDiffuzersAPICommand(
        args.output,
        args.port,
        args.host,
        args.device,
        args.workers,
    )


class RunDiffuzersAPICommand(BaseDiffuzersCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_api_parser = parser.add_parser(
            "api",
            description="✨ Run diffuzers api",
        )
        run_api_parser.add_argument(
            "--output",
            type=str,
            required=False,
            help="Output path is optional, but if provided, all generations will automatically be saved to this path.",
        )
        run_api_parser.add_argument(
            "--port",
            type=int,
            default=10000,
            help="Port to run the app on",
            required=False,
        )
        run_api_parser.add_argument(
            "--host",
            type=str,
            default="127.0.0.1",
            help="Host to run the app on",
            required=False,
        )
        run_api_parser.add_argument(
            "--device",
            type=str,
            required=False,
            help="Device to use, e.g. cpu, cuda, cuda:0, mps (for m1 mac) etc.",
        )
        run_api_parser.add_argument(
            "--workers",
            type=int,
            required=False,
            default=1,
            help="Number of workers to use",
        )
        run_api_parser.set_defaults(func=run_api_command_factory)

    def __init__(self, output, port, host, device, workers):
        self.output = output
        self.port = port
        self.host = host
        self.device = device
        self.workers = workers

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.port = str(self.port)
        self.workers = str(self.workers)

    def run(self):
        cmd = [
            "uvicorn",
            "diffuzers.api.main:app",
            "--host",
            self.host,
            "--port",
            self.port,
            "--workers",
            self.workers,
        ]

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
                print("Killing api")
                p.kill()
                p.wait()
                raise
