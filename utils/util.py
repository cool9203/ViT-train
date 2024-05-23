#! python3
# coding: utf-8

import copy
from pathlib import Path
from typing import Any, Dict, List, Union, overload

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def dict_merge(x: Dict, y: Dict) -> Dict:
    """Reference: https://stackoverflow.com/a/26853961

    Args:
        x (Dict): dict 1
        y (Dict): dict 2

    Returns:
        Dict: merged dict
    """
    z = {}
    overlapping_keys = x.keys() & y.keys()
    for key in overlapping_keys:
        if isinstance(x[key], dict) and isinstance(y[key], dict):
            z[key] = dict_merge(x[key], y[key])
        else:
            z[key] = y[key]
    for key in x.keys() - overlapping_keys:
        z[key] = copy.deepcopy(x[key])
    for key in y.keys() - overlapping_keys:
        z[key] = copy.deepcopy(y[key])
    return z


@overload
def dict_nested_get(data: Dict[str, Any], nested: Union[str, List[str]]) -> Any:
    pass


@overload
def dict_nested_get(data: Dict[str, Any], nested: Union[str, List[str]], default: Any) -> Any:
    pass


def dict_nested_get(data: Dict[str, Any], nested: Union[str, List[str]], *args, **kwds) -> Any:
    try:
        nested_list = None
        if isinstance(nested, str):
            nested_list = nested.split(".")
        elif isinstance(nested, list):
            nested_list = nested
        else:
            raise TypeError("nested need be 'str' or 'list[str]'")

        value = data
        for key in nested_list:
            if "[" in key and key[-1] == "]":
                index = int(key[key.index("[") + 1 : -1])
                key = key[: key.index("[")]
                value = value[key][index]
            else:
                value = value[key]
        return copy.deepcopy(value)

    except KeyError as e:
        if len(args) > 0:
            return args[0]
        elif "default" in kwds:
            return kwds["default"]
    raise Exception(f"Not have key of {nested}")


def load_config(
    path: Union[str, Path, List[str], List[Path]] = [
        "./pyproject.toml",
        "../pyproject.toml",
    ],
) -> Dict[str, str]:
    config = dict()
    file_path = None

    if not path:
        return config

    if isinstance(path, list):
        for p in path:
            if Path(p).exists():
                file_path = Path(p)
    else:
        file_path = Path(path)

    if file_path:
        with file_path.open("rb") as f:
            config = tomllib.load(f)

    return config


def main() -> None:
    pass


if __name__ == "__main__":
    main()
    input("press Enter to continue...")
