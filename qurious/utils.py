from typing import Optional

import pandas as pd
import torch
from IPython.display import HTML, display


def auto_device():
    """
    Automatically selects the device for PyTorch based on availability of CUDA or MPS.
    Returns:
        torch.device: The selected device (either "cuda", "mps", or "cpu").
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    print(f"using device: {device}")
    return device


def display_pd_table(dataset_or_sample, replace_newlines: Optional[list | bool] = False):
    """
    Display a pandas DataFrame or a dictionary as an HTML table in Jupyter Notebook.

    Args:
        dataset_or_sample (pd.DataFrame or dict): The dataset or sample to display.
        replace_newlines (bool or list, optional): If True, replaces newlines in string columns with <br> tags.
            If a list, only applies to the specified columns. Defaults to False.
    """
    # Set pandas options to display the DataFrame nicely
    pd.set_option("display.max_colwidth", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_rows", None)

    if isinstance(dataset_or_sample, dict):
        df = pd.DataFrame(dataset_or_sample, index=[0])
    else:
        df = pd.DataFrame(dataset_or_sample)

    # iterate over all cells in the DataFrame
    if replace_newlines:
        columns = replace_newlines if isinstance(replace_newlines, list) else df.columns
        for col in columns:
            if df[col].dtype == "object":
                # replace newlines with <br> tags
                df[col] = df[col].apply(lambda x: x.replace("\n", "<br>") if isinstance(x, str) else x)

    html = df.to_html(escape=False)
    styled_html = (
        f"""<style> .dataframe th, .dataframe tbody td {{ text-align: left; padding-right: 30px; }} </style> {html}"""
    )
    display(HTML(styled_html))
