import os
import tempfile
from unittest.mock import MagicMock, call, patch

from loguru import logger

from qurious.experiments.tracker import ConsoleTracker, FileTracker, MLflowTracker, Tracker

# filepath: qurious/experiments/test_tracker.py


class TestTracker:
    def test_tracker_base_interface(self):
        tracker = Tracker()

        # Test that all methods can be called without raising exceptions
        tracker.log_info("test message")
        tracker.log_metrics({"metric1": 1.0})
        tracker.log_metric("metric2", 2.0)
        tracker.log_artifact("some/path")
        tracker.close()

    def test_log_metric_calls_log_metrics(self):
        tracker = Tracker()
        with patch.object(tracker, "log_metrics") as mock_log_metrics:
            tracker.log_metric("key", 1.0, step=5)
            mock_log_metrics.assert_called_once_with({"key": 1.0}, step=5)


class TestConsoleTracker:
    @patch("sys.stderr")
    def test_console_tracker_init(self, mock_stderr):
        with patch.object(logger, "add") as mock_add:
            mock_add.side_effect = [1, 2]  # Return handler IDs
            tracker = ConsoleTracker("test_run")

            assert tracker.run_name == "test_run"
            assert tracker.info_logger == 1
            assert tracker.other_logger == 2
            assert mock_add.call_count == 2

    @patch("sys.stderr")
    def test_log_info(self, mock_stderr):
        with (
            patch.object(logger, "add", return_value=1),
            patch.object(logger, "contextualize") as mock_context,
            patch.object(logger, "info") as mock_info,
        ):
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock()

            tracker = ConsoleTracker("test_run")
            tracker.log_info("test message")

            mock_context.assert_called_once_with(run_name="test_run")
            mock_info.assert_called_once_with("test message")

    @patch("sys.stderr")
    def test_log_metrics(self, mock_stderr):
        with (
            patch.object(logger, "add", return_value=1),
            patch.object(logger, "contextualize") as mock_context,
            patch.object(logger, "log") as mock_log,
        ):
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock()

            tracker = ConsoleTracker("test_run")

            # Test without step
            tracker.log_metrics({"metric1": 1.0, "metric2": 2.0})
            mock_log.assert_any_call("METRIC", "metric1=1.0, metric2=2.0")

            # Test with step
            mock_log.reset_mock()
            tracker.log_metrics({"metric3": 3.0}, step=5)
            mock_log.assert_any_call("METRIC", "Step 5: metric3=3.0")

    @patch("sys.stderr")
    def test_log_artifact(self, mock_stderr):
        with (
            patch.object(logger, "add", return_value=1),
            patch.object(logger, "contextualize") as mock_context,
            patch.object(logger, "log") as mock_log,
        ):
            mock_context.return_value.__enter__ = MagicMock()
            mock_context.return_value.__exit__ = MagicMock()

            tracker = ConsoleTracker("test_run")

            # Test with default artifact_path
            tracker.log_artifact("local/path/file.txt")
            mock_log.assert_called_with("ARTIFACT", "Logged local/path/file.txt to file.txt")

            # Test with custom artifact_path
            mock_log.reset_mock()
            tracker.log_artifact("local/path/file.txt", "artifacts/file.txt")
            mock_log.assert_called_with("ARTIFACT", "Logged local/path/file.txt to artifacts/file.txt")

    @patch("sys.stderr")
    def test_close(self, mock_stderr):
        with patch.object(logger, "add") as mock_add, patch.object(logger, "remove") as mock_remove:
            mock_add.side_effect = [1, 2]  # Return handler IDs

            tracker = ConsoleTracker("test_run")
            tracker.close()

            mock_remove.assert_has_calls([call(1), call(2)])


class TestFileTracker:
    def test_file_tracker_init(self):
        with tempfile.NamedTemporaryFile() as temp:
            with patch.object(logger, "add") as mock_add:
                mock_add.side_effect = [1, 2]  # Return handler IDs
                tracker = FileTracker(temp.name)

                assert tracker.file_path == temp.name
                assert tracker.info_handler == 1
                assert tracker.other_handler == 2
                assert mock_add.call_count == 2

    def test_log_artifact(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "logs", "test.log")  # Put log file in a subdirectory
            source_file = os.path.join(temp_dir, "source.txt")

            # Create logs directory
            os.makedirs(os.path.dirname(log_file), exist_ok=True)

            # Create source file
            with open(source_file, "w") as f:
                f.write("test content")

            with patch.object(logger, "add", return_value=1):
                tracker = FileTracker(log_file)

                # Test with default artifact_path
                tracker.log_artifact(source_file)
                target_path = os.path.join(os.path.dirname(log_file), "source.txt")
                assert os.path.exists(target_path)
                with open(target_path, "r") as f:
                    assert f.read() == "test content"

                # Test with custom artifact_path
                custom_path = "artifacts/custom_name.txt"
                tracker.log_artifact(source_file, custom_path)
                target_path = os.path.join(os.path.dirname(log_file), custom_path)
                assert os.path.exists(target_path)
                with open(target_path, "r") as f:
                    assert f.read() == "test content"

    def test_close(self):
        with tempfile.NamedTemporaryFile() as temp:
            with (
                patch.object(logger, "add") as mock_add,
                patch.object(logger, "remove") as mock_remove,
                patch.object(Tracker, "close") as mock_super_close,
            ):
                mock_add.side_effect = [1, 2]  # Return handler IDs

                tracker = FileTracker(temp.name)
                tracker.close()

                mock_remove.assert_has_calls([call(1), call(2)])
                mock_super_close.assert_called_once()


class TestMLflowTracker:
    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    def test_mlflow_tracker_init(self, mock_start_run, mock_set_experiment, mock_set_uri):
        # Setup mock active run
        mock_run = MagicMock()
        mock_run.info.run_id = "test_run_id"
        mock_run.info.run_name = "test_run_name"
        mock_start_run.return_value = mock_run

        # Test init without parent_run_id
        tracker = MLflowTracker("test_experiment", "test_run_name")

        mock_set_uri.assert_called_once_with("http://127.0.0.1:5000")
        mock_set_experiment.assert_called_once_with("test_experiment")
        mock_start_run.assert_called_once_with(run_name="test_run_name")

        assert tracker.experiment_name == "test_experiment"
        assert tracker.run_id == "test_run_id"
        assert tracker.run_name == "test_run_name"
        assert tracker.active_run == mock_run

        # Test init with parent_run_id
        mock_start_run.reset_mock()
        tracker = MLflowTracker("test_experiment", "test_run_name", parent_run_id="parent_id")
        mock_start_run.assert_called_once_with(run_name="test_run_name", nested=True, parent_run_id="parent_id")

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.log_metric")
    def test_log_metric(self, mock_log_metric, mock_start_run, mock_set_experiment, mock_set_uri):
        # Setup mock active run
        mock_run = MagicMock()
        mock_start_run.return_value = mock_run

        tracker = MLflowTracker("test_experiment")
        tracker.log_metric("key", 1.0, step=5)

        mock_log_metric.assert_called_once_with("key", 1.0, 5)

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.log_metrics")
    def test_log_metrics(self, mock_log_metrics, mock_start_run, mock_set_experiment, mock_set_uri):
        # Setup mock active run
        mock_run = MagicMock()
        mock_start_run.return_value = mock_run

        tracker = MLflowTracker("test_experiment")
        tracker.log_metrics({"key1": 1.0, "key2": 2.0}, step=5)

        mock_log_metrics.assert_called_once_with({"key1": 1.0, "key2": 2.0}, 5)

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.log_artifact")
    def test_log_artifact(self, mock_log_artifact, mock_start_run, mock_set_experiment, mock_set_uri):
        # Setup mock active run
        mock_run = MagicMock()
        mock_start_run.return_value = mock_run

        tracker = MLflowTracker("test_experiment")

        # Test without artifact_path
        tracker.log_artifact("local/path/file.txt")
        mock_log_artifact.assert_called_once_with("local/path/file.txt", None)

        # Test with artifact_path
        mock_log_artifact.reset_mock()
        tracker.log_artifact("local/path/file.txt", "artifacts")
        mock_log_artifact.assert_called_once_with("local/path/file.txt", "artifacts")

    @patch("mlflow.set_tracking_uri")
    @patch("mlflow.set_experiment")
    @patch("mlflow.start_run")
    @patch("mlflow.end_run")
    def test_close(self, mock_end_run, mock_start_run, mock_set_experiment, mock_set_uri):
        # Setup mock active run
        mock_run = MagicMock()
        mock_start_run.return_value = mock_run

        tracker = MLflowTracker("test_experiment")
        tracker.close()

        mock_end_run.assert_called_once()
        assert tracker.active_run is None

        # Test close when active_run is already None
        mock_end_run.reset_mock()
        tracker.close()
        mock_end_run.assert_not_called()
