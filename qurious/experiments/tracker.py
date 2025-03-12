import os
import shutil
from datetime import datetime

import mlflow
from loguru import logger


class Tracker:
    # define log levels as enums
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    DEBUG = "debug"

    def log(self, level, message):
        pass

    def info(self, message):
        self.log(self.INFO, message)

    def warning(self, message):
        self.log(self.WARNING, message)

    def error(self, message):
        self.log(self.ERROR, message)

    def debug(self, message):
        self.log(self.DEBUG, message)

    def log_metrics(self, metrics, step=None):
        pass

    def log_metric(self, key, value, step=None):
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path, artifact_path=None):
        pass


class ConsoleTracker(Tracker):
    def log(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{level.upper()}] {timestamp} - {message}")

    def log_metrics(self, metrics, step=None):
        metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
        if step is not None:
            print(f"[METRICS] Step {step} - {metrics_str}")
        else:
            print(f"[METRICS] {metrics_str}")

    def log_artifact(self, local_path, artifact_path=None):
        if artifact_path is None:
            artifact_path = os.path.basename(local_path)
        print(f"[ARTIFACT] Logged {local_path} to {artifact_path}")


class FileTracker(Tracker):
    def __init__(self, file_path):
        self.file_path = file_path
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        self.file = open(file_path, "w")

    def log(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.file.write(f"[{level.upper()}] {timestamp} - {message}\n")
        self.file.flush()

    def log_metrics(self, metrics, step=None):
        metrics_str = " | ".join([f"{k}={v}" for k, v in metrics.items()])
        if step is not None:
            self.file.write(f"[METRICS] Step {step} - {metrics_str}\n")
        else:
            self.file.write(f"[METRICS] {metrics_str}\n")
        self.file.flush()

    def log_artifact(self, local_path, artifact_path=None):
        if artifact_path is None:
            artifact_path = os.path.basename(local_path)

        # copy file from base path to artifact path
        artifact_full_path = os.path.join(os.path.dirname(self.file_path), artifact_path)
        os.makedirs(os.path.dirname(artifact_full_path), exist_ok=True)

        shutil.copy(local_path, artifact_full_path)
        self.file.write(f"[ARTIFACT] copied {local_path} to {artifact_full_path}\n")
        self.file.flush()

    def close(self):
        if self.file:
            self.file.close()


class MLflowTracker(Tracker):
    def __init__(self, experiment_name):
        self.experiment_name = experiment_name
        self.active_run = mlflow.active_run()

    def log(self, level, message):
        # MLflow does not support logging levels directly
        pass

    def log_metric(self, key, value, step=None):
        if self.active_run:
            mlflow.log_metric(key, value, step)

    def log_metrics(self, metrics, step=None):
        if self.active_run:
            mlflow.log_metrics(metrics, step)

    def log_artifact(self, local_path, artifact_path=None):
        if self.active_run:
            mlflow.log_artifact(local_path, artifact_path)
