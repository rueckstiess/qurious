import os
import tempfile
import unittest
from datetime import datetime
from unittest import mock

# Now import the logging modules
from qurious.experiments.tracker import ConsoleTracker, FileTracker, MLflowTracker, Tracker


class TestTrackerBase(unittest.TestCase):
    """Test base Tracker class functionality."""

    def test_convenience_methods(self):
        """Test that convenience methods call log with the correct level."""
        # Create a mock of Tracker for testing
        tracker = Tracker()
        tracker.log = mock.MagicMock()

        # Call the convenience methods and verify they call log with correct args
        tracker.info("info message")
        tracker.log.assert_called_with(Tracker.INFO, "info message")

        tracker.warning("warning message")
        tracker.log.assert_called_with(Tracker.WARNING, "warning message")

        tracker.error("error message")
        tracker.log.assert_called_with(Tracker.ERROR, "error message")

        tracker.debug("debug message")
        tracker.log.assert_called_with(Tracker.DEBUG, "debug message")

    def test_metric_calls_metrics(self):
        """Test that metric calls metrics with a single key-value pair."""
        # Create a mock of Tracker for testing
        tracker = Tracker()
        tracker.log_metrics = mock.MagicMock()

        # Ensure the mock has the abstract methods implemented
        tracker.log_metrics = mock.MagicMock()

        tracker.log_metric("accuracy", 0.95, step=10)
        tracker.log_metrics.assert_called_with({"accuracy": 0.95}, step=10)


class TestConsoleTracker(unittest.TestCase):
    """Test ConsoleTracker functionality."""

    def setUp(self):
        self.console_tracker = ConsoleTracker()

    @mock.patch("builtins.print")
    def test_log(self, mock_print):
        """Test log method prints correctly formatted message."""
        with mock.patch("qurious.experiments.tracker.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 3, 12, 10, 30, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            self.console_tracker.log(Tracker.INFO, "test message")
            mock_print.assert_called_with("[INFO] 2025-03-12 10:30:00 - test message")

    @mock.patch("builtins.print")
    def test_metrics_with_step(self, mock_print):
        """Test metrics method with step prints correctly."""
        self.console_tracker.log_metrics({"accuracy": 0.95, "loss": 0.1}, step=10)

        # Since we don't know the order of the metrics in the string
        call_args = mock_print.call_args[0][0]
        self.assertTrue(call_args.startswith("[METRICS] Step 10 - "))
        self.assertTrue("accuracy=0.95" in call_args)
        self.assertTrue("loss=0.1" in call_args)

    @mock.patch("builtins.print")
    def test_metrics_without_step(self, mock_print):
        """Test metrics method without step prints correctly."""
        self.console_tracker.log_metrics({"accuracy": 0.95, "loss": 0.1})

        call_args = mock_print.call_args[0][0]
        self.assertTrue(call_args.startswith("[METRICS] "))
        self.assertTrue("accuracy=0.95" in call_args)
        self.assertTrue("loss=0.1" in call_args)

    @mock.patch("builtins.print")
    def test_artifact(self, mock_print):
        """Test artifact method prints correct message."""
        self.console_tracker.log_artifact("/path/to/local/file.txt", "artifacts/file.txt")
        mock_print.assert_called_with("[ARTIFACT] Logged /path/to/local/file.txt to artifacts/file.txt")

        # Test with default artifact path
        self.console_tracker.log_artifact("/path/to/local/file.txt")
        mock_print.assert_called_with("[ARTIFACT] Logged /path/to/local/file.txt to file.txt")


class TestFileTracker(unittest.TestCase):
    """Test FileTracker functionality."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.log_path = os.path.join(self.temp_dir.name, "logs", "test.log")
        self.file_tracker = FileTracker(self.log_path)

    def tearDown(self):
        self.file_tracker.close()
        self.temp_dir.cleanup()

    def test_log_creates_file(self):
        """Test that log method creates log file."""
        self.file_tracker.log(Tracker.INFO, "test message")
        self.assertTrue(os.path.exists(self.log_path))

    def test_log_writes_correctly(self):
        """Test that log method writes correctly formatted message."""
        with mock.patch("qurious.experiments.tracker.datetime") as mock_datetime:
            mock_datetime.now.return_value = datetime(2025, 3, 12, 10, 30, 0)
            mock_datetime.side_effect = lambda *args, **kw: datetime(*args, **kw)

            self.file_tracker.log(Tracker.INFO, "test message")

            with open(self.log_path, "r") as f:
                content = f.read()
                self.assertEqual(content, "[INFO] 2025-03-12 10:30:00 - test message\n")

    def test_metrics_with_step(self):
        """Test metrics method with step writes correctly."""
        self.file_tracker.log_metrics({"accuracy": 0.95, "loss": 0.1}, step=10)

        with open(self.log_path, "r") as f:
            content = f.read()
            self.assertTrue(content.startswith("[METRICS] Step 10 - "))
            self.assertTrue("accuracy=0.95" in content)
            self.assertTrue("loss=0.1" in content)

    def test_metrics_without_step(self):
        """Test metrics method without step writes correctly."""
        self.file_tracker.log_metrics({"accuracy": 0.95, "loss": 0.1})

        with open(self.log_path, "r") as f:
            content = f.read()
            self.assertTrue(content.startswith("[METRICS] "))
            self.assertTrue("accuracy=0.95" in content)
            self.assertTrue("loss=0.1" in content)

    def test_artifact(self):
        """Test artifact method copies file correctly."""
        # Create a test file
        test_file_path = os.path.join(self.temp_dir.name, "testfile.txt")
        with open(test_file_path, "w") as f:
            f.write("test content")

        # Log the artifact
        artifact_path = "artifacts/copied_file.txt"
        self.file_tracker.log_artifact(test_file_path, artifact_path)

        # Check that the file was copied
        expected_artifact_path = os.path.join(os.path.dirname(self.log_path), artifact_path)
        self.assertTrue(os.path.exists(expected_artifact_path))

        # Check file content
        with open(expected_artifact_path, "r") as f:
            content = f.read()
            self.assertEqual(content, "test content")

        # Check the log entry
        with open(self.log_path, "r") as f:
            log_content = f.read()
            self.assertTrue(f"[ARTIFACT] copied {test_file_path} to {expected_artifact_path}" in log_content)

    def test_artifact_default_path(self):
        """Test artifact method with default path."""
        # Create a test file
        test_file_path = os.path.join(self.temp_dir.name, "testfile.txt")
        with open(test_file_path, "w") as f:
            f.write("test content")

        # Log the artifact with default path
        self.file_tracker.log_artifact(test_file_path)

        # Check that the file was copied with default name
        expected_artifact_path = os.path.join(os.path.dirname(self.log_path), "testfile.txt")
        self.assertTrue(os.path.exists(expected_artifact_path))


class TestMLflowTracker(unittest.TestCase):
    """Test MLflowTracker functionality."""

    def setUp(self):
        self.experiment_name = "test_experiment"
        self.mlflow_tracker = MLflowTracker(self.experiment_name)

    @mock.patch("mlflow.log_metric")
    def test_metric(self, mock_log_metric):
        """Test metric method calls mlflow.log_metric."""
        with mock.patch.object(self.mlflow_tracker, "active_run", mock.MagicMock(return_value=True)):
            self.mlflow_tracker.log_metric("accuracy", 0.95, step=10)
            mock_log_metric.assert_called_with("accuracy", 0.95, 10)

    @mock.patch("mlflow.log_metrics")
    def test_metrics(self, mock_log_metrics):
        """Test metrics method calls mlflow.log_metrics."""
        with mock.patch.object(self.mlflow_tracker, "active_run", mock.MagicMock(return_value=True)):
            metrics = {"accuracy": 0.95, "loss": 0.1}
            self.mlflow_tracker.log_metrics(metrics, step=10)
            mock_log_metrics.assert_called_with(metrics, 10)

    @mock.patch("mlflow.log_artifact")
    def test_artifact(self, mock_log_artifact):
        """Test artifact method calls mlflow.log_artifact."""
        with mock.patch.object(self.mlflow_tracker, "active_run", mock.MagicMock(return_value=True)):
            self.mlflow_tracker.log_artifact("/path/to/file.txt", "artifacts")
            mock_log_artifact.assert_called_with("/path/to/file.txt", "artifacts")

    @mock.patch("mlflow.log_artifact")
    def test_artifact_default_path(self, mock_log_artifact):
        """Test artifact method with default path."""
        with mock.patch.object(self.mlflow_tracker, "active_run", mock.MagicMock(return_value=True)):
            self.mlflow_tracker.log_artifact("/path/to/file.txt")
            mock_log_artifact.assert_called_with("/path/to/file.txt", None)

    @mock.patch("mlflow.log_metric")
    def test_no_active_run(self, mock_log_metric):
        """Test that methods do nothing when no active run."""
        with mock.patch.object(self.mlflow_tracker, "active_run", None):
            self.mlflow_tracker.log_metric("accuracy", 0.95)
            mock_log_metric.assert_not_called()


if __name__ == "__main__":
    unittest.main()
