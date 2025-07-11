from box import Box
from omegaconf import DictConfig


def box_to_dict(d):
    if isinstance(d, Box):
        return d.to_dict()
    if isinstance(d, DictConfig):
        return dict(d)
    return d


def flatten_config(d, parent_key="", sep="."):
    items = []
    # Convert Box/DictConfig to dict if needed
    d = box_to_dict(d)

    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        # Convert v to dict if it's a Box or DictConfig
        v = box_to_dict(v)

        # Handle lists/tuples by converting them to strings
        if isinstance(v, (list, tuple)):
            items.append((new_key, str(v)))
        # Recursively flatten dictionaries
        elif isinstance(v, (dict, Box, DictConfig)):
            items.extend(flatten_config(v, new_key, sep=sep).items())
        # Handle all other types directly
        else:
            items.append((new_key, v))

    return dict(items)
