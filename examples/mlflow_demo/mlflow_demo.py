import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("check-localhost-connection")


with mlflow.start_run(run_name="my_run") as run:
    print(run)

    # Log parameters

    mlflow.log_param("param1", 5)
    mlflow.log_param("param2", "foo")

    # Log metrics
    for i in range(10):
        mlflow.log_metric("metric1", i, step=i)
        mlflow.log_metric("metric2", 1 / (1 + i), step=i)

    # Log file artifact
    mlflow.log_artifact("requirements.txt", artifact_path="artifacts")

    # log json artifact
    mlflow.log_dict({"key": "value"}, artifact_file="artifacts/artifact.json")

    # create fig object and log figure
    x = np.linspace(0, 10, 100)
    y = np.sin(x)
    fig, ax = plt.subplots()
    ax.plot(x, y)
    ax.set_title("Sine Wave")
    ax.set_xlabel("x")
    ax.set_ylabel("sin(x)")

    mlflow.log_figure(fig, "artifacts/sine_wave.png")

    # log table
    data = {
        "col1": [1, 2, 3],
        "col2": ["A", "B", "C"],
        "col3": [4.5, 5.5, 6.5],
    }
    df = pd.DataFrame(data)

    mlflow.log_table(df, artifact_file="table.json")
