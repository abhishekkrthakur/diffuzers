from argparse import ArgumentParser

from . import BaseDiffuzersCommand


def run_app_command_factory(args):
    return RunDiffuzersAppCommand(args.model_path, args.image_size)


class RunDiffuzersAppCommand(BaseDiffuzersCommand):
    @staticmethod
    def register_subcommand(parser: ArgumentParser):
        run_app_parser = parser.add_parser("run", description="✨ Run diffuzers app")
        run_app_parser.add_argument("--model_path", type=str, required=True, help="Path to model")
        run_app_parser.add_argument("--image_size", type=int, required=True, help="Image size")
        run_app_parser.set_defaults(func=run_app_command_factory)

    def __init__(self, model_path, image_size):
        self.model_path = model_path
        self.image_size = image_size

    def run(self):
        from ..app import Diffuzers

        app = Diffuzers(self.model_path, self.image_size).app()
        app.launch()
