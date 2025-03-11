from datasets import Dataset, DatasetDict
from datasets import load_dataset as hf_load_dataset
from pymongo import MongoClient


def load_dataset(*args, **kwargs) -> Dataset:
    """
    Wrapper for Hugging Face datasets.load_dataset() that can load data from MongoDB as well.
    If type is "mongodb", it loads data from MongoDB, otherwise it passes all arguments
    to load_dataset function from Hugging Face datasets.

    Args:
        type (str): Type of data source, "mongodb" or any of the types supported by Hugging Face datasets.
        **kwargs: Additional arguments for loading data. For MongoDB, it requires "uri", "db", and "collection",
        and optionally takes "filter", "sort", "limit", "skip", and "projection".

    Returns:
        Dataset: Loaded dataset.
    """
    path = args[0]

    if path == "mongodb":
        # Extract MongoDB connection parameters
        uri = kwargs.pop("uri", "mongodb://localhost:27017/")
        db_name = kwargs.pop("db")
        collection_name = kwargs.pop("collection")

        # Extract optional parameters with defaults
        filter_dict = kwargs.pop("filter", {})
        sort = kwargs.pop("sort", None)
        limit = kwargs.pop("limit", None)
        skip = kwargs.pop("skip", None)
        projection = kwargs.pop("projection", {"_id": 0})

        # Connect to MongoDB
        client = MongoClient(uri)
        db = client[db_name]
        coll = db[collection_name]

        # Build the query
        cursor = coll.find(filter_dict, projection)
        if sort:
            cursor = cursor.sort(sort)
        if skip:
            cursor = cursor.skip(skip)
        if limit:
            cursor = cursor.limit(limit)

        # Convert to list and create Dataset
        documents = list(cursor)
        client.close()

        return DatasetDict({"train": Dataset.from_list(documents)})
    else:
        # Use HuggingFace's load_dataset for other types
        return hf_load_dataset(*args, **kwargs)
