# llms/train_gpt2.py
import argparse

from qurious.experiments import BaseExperiment, Run


class GPT2TrainingExperiment(BaseExperiment):
    @classmethod
    def add_arguments(cls, parser):
        # First, add the base arguments
        super().add_arguments(parser)

        # Then add experiment-specific arguments if running standalone
        # These will be captured as unknown_args if run through the harness
        parser.add_argument("--myoption", type=str, help="Custom option for GPT-2 training")

    def execute(self, run: Run):
        # Log experiment start
        print("Starting GPT2 training experiment")
        print("Args:", self.args)
        print("Unknown args:", self.unknown_args)
        print("Config:\n", self.config.to_yaml())

        # Experiment implementation for run
        print("Run info:", run)
        # ...


if __name__ == "__main__":
    GPT2TrainingExperiment.main()
