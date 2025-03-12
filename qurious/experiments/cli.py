# qurious/runner.py
import argparse
import importlib
import os
import sys

from qurious.experiments import BaseExperiment


def main():
    parser = argparse.ArgumentParser(description="Qurious Experiment Runner")
    parser.add_argument("module", help="Python module path to the experiment")

    # Parse only the experiment_module first
    args, remaining = parser.parse_known_args()

    # Import the experiment module
    try:
        sys.path.append(os.getcwd())

        # Handle both "llms.train_gpt2" and "llms/train_gpt2.py" formats
        if "/" in args.module or "\\" in args.module:
            module_path = args.module.replace("/", ".").replace("\\", ".")
            if module_path.endswith(".py"):
                module_path = module_path[:-3]
        else:
            module_path = args.module

        module = importlib.import_module(module_path)

        # Find the experiment class (assuming it's the only subclass of BaseExperiment)
        experiment_class = None

        for attr_name in dir(module):
            attr = getattr(module, attr_name)
            if isinstance(attr, type) and issubclass(attr, BaseExperiment) and attr != BaseExperiment:
                experiment_class = attr
                break

        if not experiment_class:
            raise ValueError(f"No experiment class found in {args.experiment_module}")

        # Run the experiment
        experiment_class.main()

    except (ImportError, ValueError) as e:
        print(f"Error loading experiment: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
