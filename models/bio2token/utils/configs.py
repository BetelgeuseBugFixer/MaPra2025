from box import Box
from hydra_zen import load_from_yaml, builds, instantiate
from typing import get_type_hints, TypeVar, Type
from dataclasses import is_dataclass
import json
from collections.abc import Mapping

T = TypeVar("T")


def pi_builds(cls, yaml_dict: dict = {}, **kwargs):
    kwargs = recursive_merge(kwargs, yaml_dict)
    # print(f'kwargs is {kwargs}')
    main_args = get_type_hints(cls.__init__)
    # print(f'main_args is {main_args}')
    for k, v in main_args.items():
        if k in kwargs and is_dataclass(v):
            # print(f'Diving into {k}')
            kwargs[k] = pi_builds(v, **kwargs[k])
            # print(f'new kwargs is {kwargs}')
    return builds(cls, populate_full_signature=True, **kwargs)


def is_serializable(obj):
    try:
        json.dumps(obj)
        return True
    except TypeError:
        return False


def pi_instantiate(cls: Type[T], yaml_dict: dict = {}, **kwargs) -> T:
    serializable_kwargs = {k: v for k, v in kwargs.items() if is_serializable(v)}
    unserializable_kwargs = {k: v for k, v in kwargs.items() if not is_serializable(v)}
    return instantiate(pi_builds(cls, yaml_dict=yaml_dict, **serializable_kwargs), **unserializable_kwargs)


def recursive_merge(dict1, dict2):
    """
    Recursively merges dict2 into dict1 and returns a new dictionary.
    """
    result = dict1.copy()  # Make a shallow copy of dict1
    for key, value in dict2.items():
        if key in result and isinstance(result[key], Mapping) and isinstance(value, Mapping):
            result[key] = recursive_merge(result[key], value)
        else:
            result[key] = value
    return result


def utilsyaml_to_dict(yaml_file: str):
    # Step 1 - create the includes queue
    includes_queue = []
    includes_list = []
    includes_queue.append(yaml_file)
    while includes_queue:
        yaml_file = includes_queue.pop()
        includes_list.append(yaml_file)
        yaml_dict = load_from_yaml("configs/" + yaml_file)
        if "include" in yaml_dict:
            if isinstance(yaml_dict["include"], str):
                yaml_dict["include"] = [yaml_dict["include"]]
            includes_queue.extend(yaml_dict["include"])
    # reverse the list
    includes_list = includes_list[::-1]
    # Step 2 - create the config dictionary
    config_dict = {}
    for yaml_file in includes_list:
        yaml_dict = load_from_yaml("configs/" + yaml_file)
        config_dict = recursive_merge(config_dict, yaml_dict)
    if "include" in config_dict:
        del config_dict["include"]
    return Box(config_dict)
