import copy
import pickle
import torch
import io

import neptune
from pathlib import Path
from collections import defaultdict
import resource
import platform

from config.constants import Paths, Constants


class CPUUnpickler(pickle.Unpickler):
    """
    A custom Unpickler that loads objects with the CPU device.
    From: https://stackoverflow.com/questions/56369030/runtimeerror-attempting-to-deserialize-object-on-a-cuda-device
    """

    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)


def limit_memory(maxsize_GB: int) -> None:
    if maxsize_GB <= 0:
        return
    if platform.system() == "Linux":
        soft, hard = resource.getrlimit(resource.RLIMIT_RSS)
        maxsize_byte = maxsize_GB * 1024 ** 3
        resource.setrlimit(resource.RLIMIT_RSS, (maxsize_byte, hard))
        print(f"Memory limit set to {maxsize_GB}GB")


def flatten_dict(d, sep='/', parent_key='') -> dict:
    flat_dict = {}
    for k, v in d.items():
        full_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            flat_dict.update(flatten_dict(v, sep, full_key))
        else:
            flat_dict[full_key] = v
    return flat_dict


def unflatten_dict(flat_dict, sep='/'):
    nested_dict = {}
    for flat_key, value in flat_dict.items():
        keys = flat_key.split(sep)
        keys = [k for k in keys if k != ""]
        d = nested_dict
        for key in keys[:-1]:
            d = d.setdefault(key, {})
        d[keys[-1]] = value
    return nested_dict


def remove_len_zero_values(nested_dict: dict) -> dict:
    res = flatten_dict(nested_dict)
    res = {k: v for k, v in res.items() if len(v) > 0}
    return unflatten_dict(res)


def nested_key_exists(d: dict, keys: list[str]):
    if len(d) == 0:
        return False
    curr_dict = d
    for k in keys[:-1]:
        curr_dict = curr_dict.get(k, dict())
        if not isinstance(curr_dict, dict) or len(curr_dict) == 0:
            return False
    if keys[-1] in curr_dict.keys():
        return True
    else:
        return False


def rec_defaultdict():
    return defaultdict(rec_defaultdict)


def merge_dicts(src: dict, dest: dict, inplace: bool = False) -> dict:
    if not inplace:
        dest = copy.deepcopy(dest)
    # Go to the current level
    for key in src.keys():  # iterate over all keys at current level
        if (key not in dest.keys()) or (not isinstance(src[key], dict)) or (
                isinstance(src[key], dict) and not isinstance(dest[key], dict)):
            # Just directly copy if
            # 1. Key is not present in dest
            # 2. The src value is not a dict -> overwrite
            # 3. The src value is a dict, but the dest value is not -> overwrite
            dest[key] = src[key]
        else:
            # Both values are dicts -> recursive function call
            merge_dicts(src[key], dest[key], inplace=True)
    return dest


def eval_tuples_and_dicts(src_dict: dict, inplace=False) -> dict:
    if not inplace:
        src_dict = copy.deepcopy(src_dict)

    for k, v in src_dict.items():
        if isinstance(v, str) and v[0] in ["(", "["] and v[-1] in [")", "]"]:
            # Convert string to list or tuple
            src_dict[k] = eval(v)
        elif isinstance(v, dict):
            # Recursive call
            eval_tuples_and_dicts(src_dict[k], inplace=True)
        elif v == "None":
            src_dict[k] = None

    return src_dict


def load_objects(config: dict, inplace=True, runs=None, root_path: Path = None) -> dict:
    log_prefix = config.pop("log_prefix", "")
    if not inplace:
        config = copy.deepcopy(config)
    if runs is None:
        runs = dict()
    if root_path is None:
        root_path = Paths.TMP
    keys = list(config.keys())
    for k in keys:
        v = config[k]
        if k == "load_objects":
            config["loaded_objects"] = dict()
            run_id = v["from_id"]
            obj_names = v["names"]

            for obj_name in obj_names:
                file_path = (root_path / run_id / f"{obj_name}.pkl")
                if not file_path.is_file():
                    file_path.parent.mkdir(exist_ok=True, parents=True)
                    # Download file, if it's not downloaded yet
                    if run_id in runs:
                        # Create run, if not created yet
                        run = runs[run_id]
                    else:
                        run = neptune.init_run(
                            project=Constants.NEPTUNE_LOGGER_PROJECT,
                            api_token=Constants.NEPTUNE_LOGGER_API_TOKEN,
                            with_id=run_id,
                        )
                        runs[run_id] = run
                    run[f"{log_prefix}saved_objects/{obj_name}.pkl"].download(str(file_path))
                # Load pickle file from disk
                with open(file_path, "rb") as f:
                    obj = pickle.load(f)
                config["loaded_objects"][obj_name] = obj
        elif isinstance(v, dict):
            load_objects(v, inplace=True, runs=runs, root_path=root_path)

    # In the end, stop all opened runs
    for run_id, run in runs.items():
        run.stop()

    return config


def value_lists_to_value(d: dict):
    d = flatten_dict(d)
    for k, v in d.items():
        try:
            if len(v) == 1:
                d[k] = v[0]
        except:
            pass
    return unflatten_dict(d)


def parse_name(name: str) -> dict:
    """
    Parse a string like "key1=value1_key2=value2_key3" to a dictionary
    :param name: The name string to be parsed
    :return: The dictionary, for example {"key1": "value1", "key2": "value2", "key3": True}
    """
    parts = name.split("_")
    res = {}
    for part in parts:
        if "=" in part:
            key, value = part.split("=")
            res[key] = value
        else:
            res[part] = True
    return res
