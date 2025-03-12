import datetime
import os
import uuid

import mlflow

from qurious.experiments.tracker import ConsoleTracker, FileTracker, MLflowTracker, Tracker


class Run(Tracker):
    def __init__(self, experiment_name, config, run_name=None, parent_run_id=None, log_to=None):
        self.experiment_name = experiment_name
        self.config = config
        self.run_id = None
        self.run_name = run_name
        self.parent_run_id = parent_run_id
        self.log_to = log_to or ["console"]
        self.metrics = {}
        self.artifacts = {}
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

    def log(self, level, message):
        for logger in self.loggers:
            getattr(logger, level)(message)
        return self

    def log_metrics(self, metrics, step=None):
        """Log a metric to all loggers"""
        for logger in self.loggers:
            logger.log_metrics(metrics, step)
        return self

    def log_artifact(self, local_path, artifact_path=None):
        """Log an artifact to all loggers"""
        for logger in self.loggers:
            logger.log_artifact(local_path, artifact_path)
        return self

    def _setup_logging(self):
        """Setup logging based on args.log_to"""

        self.loggers = []

        if "console" in self.log_to:
            self.loggers.append(ConsoleTracker())
        if "file" in self.log_to:
            log_dir = f"logs/{self.experiment_name}/{self.run_name}"
            os.makedirs(log_dir, exist_ok=True)
            self.loggers.append(FileTracker(f"{log_dir}/run.log"))
        if "mlflow" in self.log_to:
            self.loggers.append(MLflowTracker(self.experiment_name))

        pass

    def start(self):
        """Start the run and set up logging"""

        self._setup_logging()
        self.start_time = datetime.datetime.now()
        self.is_running = True

        # Set up MLflow tracking if needed
        if "mlflow" in self.loggers:
            mlflow.set_tracking_uri("http://127.0.0.1:5000")
            mlflow.set_experiment(self.experiment_name)

            # Start run with optional parent
            if self.parent_run_id:
                active_run = mlflow.start_run(run_name=self.run_name, nested=True, parent_run_id=self.parent_run_id)
            else:
                active_run = mlflow.start_run(run_name=self.run_name)

            self.run_id = active_run.info.run_id

            # Log config parameters
            if hasattr(self.config, "flatten_and_stringify"):
                mlflow.log_params(self.config.flatten_and_stringify())
        else:
            # if mlflow is not used, create our own run_id as a UUID
            self.run_id = str(uuid.uuid4().hex)
            self.run_name = self.run_name or f"run-{self.run_id[:8]}"

        self.info(f"Starting{' nested' if self.parent_run_id else ''} run {self.run_name} (ID: {self.run_id})")
        return self

    def end(self):
        """End the run and clean up resources"""
        self.is_running = False
        self.end_time = datetime.datetime.now()

        self.info(f"Ending run {self.run_name}")

        # Close MLflow run if active
        if "mlflow" in self.loggers and self.run_id:
            mlflow.end_run()

        return self
