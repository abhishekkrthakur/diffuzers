from argparse import ArgumentParser

from . import BaseDiffuzersCommand


def run_app_command_factory(args):
    return RunDiffuzersAppCommand(args.model, args.output_path, args.share, args.port, args.host)


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
            help="Path to model",
        )
        run_app_parser.add_argument(
            "--output_path",
            type=str,
            required=True,
            help="Output path",
        )
        run_app_parser.add_argument(
            "--share",
            action="store_true",
            help="Share the app",
        )
        run_app_parser.add_argument(
            "--port",
            type=int,
            default=7860,
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
        run_app_parser.set_defaults(func=run_app_command_factory)

    def __init__(self, model, output_path, share, port, host):
        self.model = model
        self.output_path = output_path
        self.share = share
        self.port = port
        self.host = host

    def run(self):
        from ..app import Diffuzers

        print(self.share)
        app = Diffuzers(self.model, self.output_path).app()
        app.launch(show_api=False, share=self.share, server_port=self.port, server_name=self.host)
