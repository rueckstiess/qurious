from loguru import logger

from qurious.experiments import BaseExperiment, Run


class GPT2TrainingExperiment(BaseExperiment):
    """
    This is an example of a custom experiment class to demonstate the experiment wrapping."

    Call directly with:
        python qurious/experiments/example.py --myoption my_value

    Or run through the harness with:
        python qurious/experiments/cli.py qurious.experiments.example --myoption my_value

    To create your own experiments, derive from BaseExperiment and implement the execute method.
    You can optionally override the add_arguments method to add custom command line arguments.
    """

    @classmethod
    def add_arguments(cls, parser):
        # First, add the base arguments
        super().add_arguments(parser)

        # Then add experiment-specific arguments if running standalone
        # These will be captured as unknown_args if run through the harness
        parser.add_argument("--myoption", type=str, help="Custom option for GPT-2 training")

    def execute(self, run: Run):
        # Log experiment start
        logger.info("General log message unrelated to run.")
        run.log_info("Log message from example.py but related to run.")
        loss = 0.1234
        accuracy = 0.9876
        run.log_metrics({"loss": loss, "accuracy": accuracy})
        logger.warning("This is a warning message")


if __name__ == "__main__":
    GPT2TrainingExperiment.main()
