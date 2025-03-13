import os
import shutil
import sys
from typing import Optional

import mlflow
from loguru import logger

RUN_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</> | <lvl>{level: <8}</> | <lvl>{message}</>"
OTHER_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</> | <lvl>{level: <8}</> | <cyan>{name}:{function}():{line}</> | <lvl>{message}</>"
DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}()</cyan>:<cyan>{line}</cyan> | "
    "<level>{message}</level>"
)

logger.remove()
logger.add(sys.stderr, format=DEFAULT_FORMAT, filter=lambda record: "run_name" not in record["extra"], level="INFO")
logger.level("INFO", color="<white>")
logger.level("METRIC", no=20, color="<white><b>")
logger.level("ARTIFACT", no=20, color="<fg #FF8300>")


class Tracker:
    def log_info(self, message):
        pass

    def log_metrics(self, metrics, step=None):
        pass

    def log_metric(self, key, value, step=None):
        self.log_metrics({key: value}, step=step)

    def log_artifact(self, local_path, artifact_path=None):
        pass

    def close(self):
        pass


class ConsoleTracker(Tracker):
    def __init__(self, run_name):
        self.run_name = run_name

        RUN_FORMAT = "<green>{time:YYYY-MM-DD HH:mm:ss}</> | <lvl>{level: <8}</> | <magenta>{extra[run_name]}</> | <lvl>{message}</>"

        self.info_logger = logger.add(
            sys.stderr,
            format=RUN_FORMAT,
            filter=lambda record: record["extra"].get("run_name", None) == run_name and record["level"].no == 20,
        )
        self.other_logger = logger.add(
            sys.stderr,
            format=OTHER_FORMAT,
            filter=lambda record: record["extra"].get("run_name", None) == run_name and record["level"].no != 20,
        )

    @staticmethod
    def _format_metric(name, value):
        if name in ["lr", "learning_rate"]:
            return f"{name}: {value:.2e}"
        elif isinstance(value, int):
            return f"{name}: {value}"
        elif "acc" in name:
            return f"{name}: {value:.2%}"
        elif isinstance(value, float):
            return f"{name}: {value:.4f}"
        else:
            return f"{name}: {value}"

    def log_info(self, message):
        with logger.contextualize(run_name=self.run_name):
            logger.info(message)

    def log_metrics(self, metrics, step=None):
        metrics_str = ", ".join([f"{self._format_metric(k, v)}" for k, v in metrics.items()])
        with logger.contextualize(run_name=self.run_name):
            if step is not None:
                logger.log("METRIC", f"Step {step}: {metrics_str}")
            else:
                logger.log("METRIC", f"{metrics_str}")

    def log_artifact(self, local_path, artifact_path=None):
        if artifact_path is None:
            artifact_path = os.path.basename(local_path)
        with logger.contextualize(run_name=self.run_name):
            logger.log("ARTIFACT", f"Logged {local_path} to {artifact_path}")

    def close(self):
        logger.remove(self.info_logger)
        logger.remove(self.other_logger)


class FileTracker(Tracker):
    def __init__(self, file_path):
        self.file_path = file_path
        self.info_handler = logger.add(file_path, format=RUN_FORMAT, filter=lambda record: record["level"].no == 20)
        self.other_handler = logger.add(file_path, format=OTHER_FORMAT, filter=lambda record: record["level"].no != 20)

    def log_metrics(self, metrics, step=None):
        # automatically logged from regular logger
        pass

    def log_artifact(self, local_path, artifact_path=None):
        if artifact_path is None:
            artifact_path = os.path.basename(local_path)

        # copy file from base path to artifact path
        artifact_full_path = os.path.join(os.path.dirname(self.file_path), artifact_path)
        os.makedirs(os.path.dirname(artifact_full_path), exist_ok=True)

        shutil.copy(local_path, artifact_full_path)

    def close(self):
        logger.remove(self.info_handler)
        logger.remove(self.other_handler)
        return super().close()


class MLflowTracker(Tracker):
    def __init__(self, experiment_name, run_name: Optional[str] = None, parent_run_id: Optional[str] = None):
        self.experiment_name = experiment_name
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(self.experiment_name)

        # Start run with optional parent
        if parent_run_id:
            active_run = mlflow.start_run(run_name=run_name, nested=True, parent_run_id=parent_run_id)
        else:
            active_run = mlflow.start_run(run_name=run_name)

        self.run_id = active_run.info.run_id
        self.run_name = active_run.info.run_name
        self.active_run = active_run

    def log_metric(self, key, value, step=None):
        if self.active_run:
            mlflow.log_metric(key, value, step)

    def log_metrics(self, metrics, step=None):
        if self.active_run:
            mlflow.log_metrics(metrics, step)

    def log_artifact(self, local_path, artifact_path=None):
        if self.active_run:
            mlflow.log_artifact(local_path, artifact_path)

    def close(self):
        if self.active_run:
            mlflow.end_run()
        self.active_run = None
