import datetime
import os
import uuid
from pathlib import Path

import mlflow
import yaml
from loguru import logger

from qurious.experiments.tracker import ConsoleTracker, FileTracker, MLflowTracker


class Run:
    def __init__(self, experiment_name, config, run_name=None, parent_run_id=None, log_to=None):
        self.experiment_name = experiment_name
        self.config = config
        self.run_id = None
        self.run_name = run_name
        self.run_path = None
        self.parent_run_id = parent_run_id
        self.log_to = log_to or []
        self.start_time = None
        self.end_time = None
        self.is_running = False

    def __enter__(self):
        """Context manager entry"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.end()
        # Return False to propagate exceptions
        return False

    def __repr__(self):
        """String representation of the run"""
        details = {
            "experiment_name": self.experiment_name,
            "run_name": self.run_name,
            "run_id": self.run_id,
            "nested": self.parent_run_id is not None,
            "parent_run_id": self.parent_run_id,
            "is_running": self.is_running,
        }
        return f"Run({', '.join(f'{k}={v}' for k, v in details.items())})"

    def to_dict(self):
        """Convert the run to a dictionary"""
        return {
            "config": self.config.to_dict(),
            "experiment_name": self.experiment_name,
            "run_path": self.run_path,
            "run_name": self.run_name,
            "run_id": self.run_id,
            "parent_run_id": self.parent_run_id,
            "nested": self.parent_run_id is not None,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "log_to": self.log_to,
            "is_running": self.is_running,
        }

    def log_info(self, message):
        """Log an info message to all trackers"""
        for tracker in self.trackers:
            tracker.log_info(message)
        return self

    def log_metrics(self, metrics, step=None):
        """Log a metric to all trackers"""
        for tracker in self.trackers:
            tracker.log_metrics(metrics, step)
        return self

    def log_artifact(self, local_path, artifact_path=None):
        """Log an artifact to all trackers"""
        for tracker in self.trackers:
            tracker.log_artifact(local_path, artifact_path)
        return self

    def _save_run_info(self):
        """Save run information to a YAML file"""
        with open(Path(self.run_path) / "run_info.yaml", "w") as outfile:
            yaml.dump(self.to_dict(), outfile, default_flow_style=False)

    def _setup_logging(self):
        global logger

        """Setup logging based on args.log_to"""

        self.trackers = []

        if "mlflow" in self.log_to:
            # First check if MLflow is enabled, which gives us the run_name and run_id
            self.trackers.append(MLflowTracker(self.experiment_name, self.run_name, self.parent_run_id))
            self.run_id = mlflow.active_run().info.run_id
            self.run_name = mlflow.active_run().info.run_name
            # Log config parameters
            mlflow.log_params(self.config.flatten_and_stringify())
            if self.parent_run_id:
                parent_run_name = mlflow.get_run(self.parent_run_id).info.run_name
                self.run_path = os.path.abspath(
                    f"./runs/{self.experiment_name}/{parent_run_name}/subruns/{self.run_name}"
                )
            else:
                self.run_path = os.path.abspath(f"./runs/{self.experiment_name}/{self.run_name}")

        else:
            # If MLflow is not used, generate a unique run_id and set the run_name
            self.run_id = str(uuid.uuid4().hex)
            self.run_name = self.run_name or f"{self.run_id[:8]}"
            if self.parent_run_id:
                parent_run_name = self.parent_run_id[:8]
                self.run_path = os.path.abspath(
                    f"./runs/{self.experiment_name}/{parent_run_name}/subruns/{self.run_name}"
                )
            else:
                self.run_path = os.path.abspath(f"./runs/{self.experiment_name}/{self.run_name}")

        # Set up other loggers based on the log_to argument
        os.makedirs(self.run_path, exist_ok=True)

        if "file" in self.log_to:
            self.trackers.append(FileTracker(f"{self.run_path}/run.log"))

        if "console" in self.log_to:
            self.trackers.append(ConsoleTracker(self.run_name))

        # Bind the run_name to the logger so that all logger.* calls include the run_name
        logger = logger.bind(run_name=self.run_name)

    def start(self):
        """Start the run and set up logging"""

        self.start_time = datetime.datetime.now()
        self.is_running = True

        self._setup_logging()
        self._save_run_info()

        logger.info(f"=== Starting{' nested' if self.parent_run_id else ''} run {self.run_name} ===")
        return self

    def end(self):
        """End the run and clean up resources"""
        self.is_running = False
        self.end_time = datetime.datetime.now()

        logger.info(f"=== Ending{' nested' if self.parent_run_id else ''} run {self.run_name} ===")

        # Close all trackers
        for tracker in self.trackers:
            tracker.close()

        self._save_run_info()

        return self
