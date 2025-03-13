import argparse
import os
from abc import ABC, abstractmethod

import mlflow

from qurious.config import Config, ConfigProduct
from qurious.experiments import Run


class BaseExperiment(ABC):
    def __init__(self, config: Config, args: argparse.Namespace = None, unknown_args: list = None):
        """Initialize the experiment with the experiment name"""
        super().__init__()
        self.experiment_name = args.experiment
        self.config = config
        self.args = args
        self.unknown_args = unknown_args

    @abstractmethod
    def execute(self, run: Run):
        """Execute the experiment for a run. Must be implemented by subclasses."""
        pass

    def execute_run(self, config: Config, parent_run_id=None, run_name=None):
        """Run the experiment with a Run object"""
        self.setup()

        # Create a Run object
        run = Run(
            experiment_name=self.experiment_name,
            config=config,
            run_name=run_name,
            parent_run_id=parent_run_id,
            log_to=self.args.log_to,
        )

        # Execute within the run context (calls run.start() and run.end())
        with run:
            try:
                self.execute(run)
            except Exception as e:
                # Handle exceptions during execution
                run.error(f"Error during execution: {e}")
                self.teardown()
                raise e

        self.teardown()

    def setup(self):
        """Setup the experiment"""
        pass

    def teardown(self):
        """Clean up resources"""
        if mlflow.active_run():
            mlflow.end_run()

    def execute_runs_from_config(self):
        # Handle multi-config case
        multi_config = ConfigProduct(self.config)

        if len(multi_config) == 1:
            self.execute_run(self.config, run_name=self.args.run_name)

        else:
            toplevel_run = Run(
                experiment_name=self.experiment_name,
                config=self.config,
                run_name=self.args.run_name,
                parent_run_id=None,
                log_to=self.args.log_to,
            )

            with toplevel_run:
                parent_run_id = toplevel_run.run_id

                for run_config in multi_config:
                    # Extract run name from config differences
                    diff = run_config.diff(self.config)
                    run_name = ", ".join([f"{k.split('.')[-1]}={v[0]}" for k, v in diff.items()])
                    self.execute_run(run_config, parent_run_id=parent_run_id, run_name=run_name)

    @classmethod
    def add_arguments(cls, parser):
        """Add base arguments to the parser"""
        parser.add_argument("--config", "-c", type=str, default=None, help="Path to the config file.")
        parser.add_argument("--params", "-p", nargs="+", default=[], help="Parameters to override in the config file.")
        parser.add_argument("--experiment", "-e", type=str, default="default-experiment", help="Experiment name.")
        parser.add_argument("--run-name", "-n", type=str, default=None, help="Run name.")
        parser.add_argument("--debug", action="store_true", help="Enable debug mode")
        parser.add_argument(
            "--log-to", nargs="+", default=["console"], choices=["console", "file", "mlflow"], help="Loggers to use."
        )

    @classmethod
    def create_parser(cls):
        """Create argument parser with base arguments"""
        parser = argparse.ArgumentParser(description="Base experiment framework")
        cls.add_arguments(parser)
        return parser

    @classmethod
    def parse_args_and_config(cls, args=None):
        """Parse arguments and create config"""
        parser = cls.create_parser()
        args, unknown = parser.parse_known_args(args)
        config = Config.from_args(args)
        return args, unknown, config

    @classmethod
    def main(cls):
        """Main entry point for standalone execution"""

        # Parse args and create config
        args, unknown_args, config = cls.parse_args_and_config()

        # Create experiment instance and execute runs
        experiment = cls(config=config, args=args, unknown_args=unknown_args)
        experiment.execute_runs_from_config()


if __name__ == "__main__":
    print("This module is not meant to be run directly. Use the command line interface.")
