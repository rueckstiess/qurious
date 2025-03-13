from qurious.config import Config
from qurious.experiments import Run


class TestRun:
    def test_run_init(self):
        """Test the initialization of the Run class."""
        run = Run(experiment_name="test_experiment", config=Config(), run_name="test_run")

        assert run.experiment_name == "test_experiment"
        assert isinstance(run.config, Config)
        assert run.run_name == "test_run"
        assert run.run_id is None
        assert run.parent_run_id is None
        assert run.log_to == []
        assert run.start_time is None
        assert run.end_time is None
        assert not run.is_running

    def test_run_init_with_parent(self):
        """Test the initialization of the Run class with a parent run ID."""
        parent_run_id = "parent_run_id"
        run = Run(
            experiment_name="test_experiment",
            config=Config(),
            run_name="test_run",
            parent_run_id=parent_run_id,
        )

        assert run.experiment_name == "test_experiment"
        assert isinstance(run.config, Config)
        assert run.run_name == "test_run"
        assert run.run_id is None
        assert run.parent_run_id == parent_run_id
        assert run.log_to == []
        assert run.start_time is None
        assert run.end_time is None
        assert not run.is_running

    def test_run_repr(self):
        """Test the string representation of the Run class."""
        run = Run(experiment_name="test_experiment", config=Config(), run_name="test_run")

        expected_repr = (
            "Run(experiment_name=test_experiment, run_name=test_run, run_id=None, "
            "nested=False, parent_run_id=None, is_running=False)"
        )
        assert repr(run) == expected_repr
        assert run.__str__() == expected_repr

    def test_run_context_manager(self):
        """Test the context manager functionality of the Run class."""
        run = Run(experiment_name="test_experiment", config=Config(), run_name="test_run")

        assert not run.is_running
        assert run.start_time is None

        with run:
            assert run.is_running
            assert run.start_time is not None

        assert not run.is_running
        assert run.end_time is not None
        assert run.end_time > run.start_time

    def test_run_start_end_explicit(self):
        """Test the explicit start method of the Run class."""
        run = Run(experiment_name="test_experiment", config=Config(), run_name="test_run")

        assert not run.is_running
        assert run.start_time is None

        run.start()

        assert run.is_running
        assert run.start_time is not None

        # Simulate ending the run
        run.end()

        assert not run.is_running
        assert run.end_time is not None
        assert run.end_time > run.start_time

    def test_run_with_parent(self):
        """Test the context manager functionality of the Run class with a parent run."""
        parent_run = Run(experiment_name="test_experiment", config=Config(), run_name="parent_run")

        assert not parent_run.is_running
        assert parent_run.start_time is None

        with parent_run:
            assert parent_run.is_running
            assert parent_run.start_time is not None

            child_run = Run(
                experiment_name="test_experiment",
                config=Config(),
                run_name="child_run",
                parent_run_id=parent_run.run_id,
            )

            assert not child_run.is_running

            with child_run:
                assert child_run.is_running
                assert child_run.start_time is not None
                assert child_run.parent_run_id == parent_run.run_id

        assert not parent_run.is_running
        assert parent_run.end_time is not None
        assert parent_run.end_time >= parent_run.start_time
        assert not child_run.is_running
        assert child_run.end_time is not None
        assert child_run.end_time >= child_run.start_time
