import copy
import math
import shutil
from typing import Type

import lightning.pytorch as pl
import ray
from ray import tune, air
from ray.tune import CLIReporter
from ray.tune.schedulers import MedianStoppingRule, FIFOScheduler
from ray.tune.search import BasicVariantGenerator
from ray.tune.search.hyperopt import HyperOptSearch

from config.constants import Paths, Constants
from data.data_processing import DataUtils
from training.training import train_func
from utils.misc import flatten_dict, unflatten_dict


def find_tune_keys(d, parent_key='', sep='.') -> list[str]:
    """Recursively find all keys containing 'ray.tune' in a nested dictionary."""
    keys = []
    for k, v in d.items():
        full_key = f"{parent_key}{sep}{k}" if parent_key else k
        if "ray.tune" in str(type(v)):
            keys.append(full_key)
        elif isinstance(v, dict) and len(v) == 1 and "grid_search" in v:
            keys.append(full_key)
        if isinstance(v, dict):
            keys.extend(find_tune_keys(v, full_key, sep=sep))
    return keys


def get_scheduler(name, **kwargs):
    if name == "MedianStoppingRule":
        return MedianStoppingRule(
            time_attr="training_iteration",
            hard_stop=True,
            **kwargs,
        )
    elif name == "FIFOScheduler":
        return FIFOScheduler(**kwargs)


def get_search_alg(name, **kwargs):
    if name in ["HyperOptSearch", "hyperopt", "HyperOpt", "Hyperopt"]:
        return HyperOptSearch(**kwargs)
    elif name in ["random", "Random", "RandomSearch", "BasicVariantGenerator"]:
        return BasicVariantGenerator(**kwargs)
    else:
        raise ValueError("Search algorithm must be 'HyperOptSearch' or 'RandomSearch'.")


def sample_config(config: dict, take_upper: bool = False, take_lower: bool = False) -> dict:
    assert not (take_lower and take_upper)
    flattened_config = flatten_dict(copy.deepcopy(config), sep=".")
    for k, v in flattened_config.items():
        if "ray.tune" in str(type(v)):
            if take_upper:
                flattened_config[k] = v.upper - 1 if isinstance(v.upper, int) else v.upper
            elif take_lower:
                flattened_config[k] = v.lower if isinstance(v.lower, int) else v.lower
            else:
                flattened_config[k] = v.sample()
        else:
            flattened_config[k] = v
    return unflatten_dict(flattened_config, sep=".")


def get_best_config(tune_config: dict):
    restored_tuner = tune.Tuner.restore(str(Paths.RAY_RESULTS / tune_config["name"]), trainable=train_func)
    result_grid = restored_tuner.get_results()
    assert result_grid.num_errors == 0, "There are errors in the results grid."
    tune_experiment_analysis = ray.tune.ExperimentAnalysis(
        str(Paths.RAY_RESULTS / tune_config["name"]),
        default_metric=tune_config["training"]["objective"][0],
        default_mode=tune_config["training"]["objective"][1],
    )
    best_config = tune_experiment_analysis.get_best_config()
    return best_config


def run_tune(
        config: dict,
        accelerator: str,
        datamodule_class: Type[pl.LightningDataModule],
        keep_ray_results: bool = False):
    name = config["name"]
    pl.seed_everything(config.get("seed", Constants.DEFAULT_SEED))

    if "objective" not in config["ray_tune"]:
        config["ray_tune"]["objective"] = config["training"]["objective"]
    objective_metric_tune, mode_tune = config["ray_tune"]["objective"]

    ray.init(
        include_dashboard=False,
        object_store_memory=int(4e9) if accelerator == "gpu" else int(2e9),
        num_cpus=config["ray_tune"]["total_cpus"],
    )

    scheduler = get_scheduler(**config["ray_tune"]["scheduler"])

    search_alg_config = config["ray_tune"].get("search_alg", {"name": "random"})
    search_alg_config["metric"] = objective_metric_tune
    search_alg_config["mode"] = mode_tune
    search_alg = get_search_alg(**search_alg_config)

    resources_per_trial = {
        "CPU": math.floor(config["ray_tune"]["total_cpus"] / config["ray_tune"]["max_concurrent_trials"]),
        "GPU": (1 / config["ray_tune"]["max_concurrent_trials"]) if accelerator == "gpu" else 0
    }

    # parameter_columns = [k for k, v in config.items() if "ray.tune" in str(type(v))]
    # parameter_columns = find_tune_keys(config, sep="/")

    reporter = CLIReporter(
        # parameter_columns=parameter_columns,
        metric_columns=[objective_metric_tune]
    )

    # data_module_manager = DataModuleManager()

    config["dataset"]["git_hash"] = DataUtils.get_git_hash()
    if find_tune_keys(config["dataset"]) == []:
        print("Didn't find tune keys in dataset config. Setting up data module for all trials in advance.")
        data_module = datamodule_class(config["dataset"], num_workers=Constants.NUM_WORKERS)
        data_module.setup(stage="fit")
        print("Data module setup complete.")
    else:
        print("There are tune keys in dataset config. Setting up data module for each trial separately.")
        data_module = None

    train_fn_with_parameters = tune.with_parameters(train_func,
                                                    with_tune=True,
                                                    data_module=data_module,
                                                    )
    if config["ray_tune"]["restore"]:
        tuner = tune.Tuner.restore(str(Paths.RAY_RESULTS / name),
                                   trainable=tune.with_resources(
                                       train_fn_with_parameters,
                                       resources=resources_per_trial
                                   ),
                                   restart_errored=True)
    else:
        tuner = tune.Tuner(
            tune.with_resources(
                train_fn_with_parameters,
                resources=resources_per_trial
            ),
            tune_config=tune.TuneConfig(
                metric=config["ray_tune"]["objective"][0],
                mode=config["ray_tune"]["objective"][1],
                scheduler=scheduler,
                num_samples=config["ray_tune"]["num_samples"],
                max_concurrent_trials=config["ray_tune"]["max_concurrent_trials"],
                search_alg=search_alg,
                reuse_actors=False,
            ),
            run_config=air.RunConfig(
                name=name,
                progress_reporter=reporter,
                storage_path=Paths.RAY_RESULTS,
            ),
            param_space=config,
        )

    results = tuner.fit()
    if not config.get("debug", False):
        # with open(str(Paths.RAY_RESULTS / name / "results.pkl"), "wb") as file:
        #     pickle.dump(results, file)
        results_df = results.get_dataframe()
        results_df.to_pickle(str(Paths.RAY_RESULTS / name / "results_df.pkl"))
        print("Best hyperparameters found where: ",
              results.get_best_result(metric=objective_metric_tune, mode=mode_tune, scope="all").config)
    ray.shutdown()

    if not keep_ray_results:
        # delete the results folder
        shutil.rmtree(str(Paths.RAY_RESULTS / name))
