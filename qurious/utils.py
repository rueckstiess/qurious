from typing import Callable, Mapping, Optional, Union

import pandas as pd
import torch
from IPython.display import HTML, display
from rich import box
from rich.table import Table


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


def walk_all_leaf_kvs(
    item,
    path="",
    parent: Optional[Union[list, Mapping]] = None,
    idx: Union[int, str] = None,
    key: str = None,
    pos: int = None,
    include_pos_in_path: bool = False,
):
    """
    Recursively walks through a nested dictionary or list structure and yields all the leaf key-value pairs.

    Args:
        item (Union[dict, OrderedDict, list]): The item to walk through.
        path (str, optional): The current path in the nested structure. Defaults to "".
        parent (Any, optional): The parent of the current item. Defaults to None.
        idx (Union[int, None], optional): The index of the current item in the list. Defaults to None.
        include_pos_in_path (bool, optional): Whether to include the position of the item in the path. Defaults to False.
            if True: {foo: [{bar: "x"}]} will produce path "foo.[0].bar" for value "x"
            if False: {foo: [{bar: "x"}]} will produce path "foo.[].bar" for value "x"

    Yields:
        dict: A dictionary with the following keys:
            - parent (Any): The parent of the current item.
            - idx (Union[int, str]): The index of the current item in the parent. int for lists, str for dicts.
            - key (str): The key of the current dictionary item.
            - pos (int): The position of the current list item.
            - value (Any): The value of the current item.
            - path (str): The current path in the nested structure.
    """
    if isinstance(item, Mapping):
        for key, value in item.items():
            new_path = f"{path}.{key}".lstrip(".")
            yield from walk_all_leaf_kvs(
                value, path=new_path, parent=item, idx=key, key=key, pos=pos, include_pos_in_path=include_pos_in_path
            )
    elif isinstance(item, list):
        for pos, value in enumerate(item):
            if include_pos_in_path:
                new_path = f"{path}.[{pos}]".lstrip(".")
            else:
                new_path = f"{path}.[]".lstrip(".")
            yield from walk_all_leaf_kvs(
                value,
                path=new_path,
                parent=item,
                idx=pos,
                pos=pos,
                key=key,
                include_pos_in_path=include_pos_in_path,
            )
    else:
        yield {"parent": parent, "idx": idx, "key": key, "pos": pos, "value": item, "path": path}


def count_parameters(m: torch.nn.Module, only_trainable: bool = False):
    """
    Returns the total number of parameters used by `m` (only counting
    shared parameters once); if `only_trainable` is True, then only
    includes parameters with `requires_grad = True`
    """
    parameters = list(m.parameters())
    if only_trainable:
        parameters = [p for p in parameters if p.requires_grad]
    unique = {p.data_ptr(): p for p in parameters}.values()
    return sum(p.numel() for p in unique)


def flatten_dict(d: dict) -> list[dict]:
    flat = {item["path"]: item["value"] for item in walk_all_leaf_kvs(d, include_pos_in_path=True)}

    return flat


def process_leaf_values(data: dict, fn: Callable) -> dict:
    """Process all leaf values in a config according to function fn."""
    if isinstance(data, dict):
        return {k: process_leaf_values(v, fn) for k, v in data.items()}
    else:
        return fn(data)


def df_to_rich_table(
    pandas_dataframe: pd.DataFrame,
    show_index: bool = True,
    index_name: Optional[str] = None,
) -> Table:
    """Convert a pandas.DataFrame obj into a rich.Table obj.
    Args:
        pandas_dataframe (DataFrame): A Pandas DataFrame to be converted to a rich Table.
        rich_table (Table): A rich Table that should be populated by the DataFrame values.
        show_index (bool): Add a column with a row count to the table. Defaults to True.
        index_name (str, optional): The column name to give to the index column. Defaults to None, showing no value.
    Returns:
        Table: The rich Table instance passed, populated with the DataFrame values."""

    rich_table = Table(show_header=True, header_style="bold magenta")

    rich_table.row_styles = ["none", "dim"]
    rich_table.box = box.SIMPLE_HEAD

    if show_index:
        index_name = str(index_name) if index_name else ""
        rich_table.add_column(index_name)

    for column in pandas_dataframe.columns:
        rich_table.add_column(str(column))

    for index, value_list in enumerate(pandas_dataframe.values.tolist()):
        row = [str(index)] if show_index else []
        row += [str(x) for x in value_list]
        rich_table.add_row(*row)

    return rich_table
