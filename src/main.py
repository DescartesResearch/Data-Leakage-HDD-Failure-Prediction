import sys
from collections import defaultdict
from pathlib import Path

import pandas as pd

from utils.paper_figures_tables import generate_tables_and_diagrams

# Add the src directory to the path to be able to import the modules
sys.path.append(str(Path(__file__).parent.resolve()))

import torch
import argparse

from data.data_module import HDDDataModule
from training.logging import get_logger, aggregate_versions
from training.training import train_func, test_func, update_config
from utils.misc import limit_memory
from utils.ray_utils import find_tune_keys, run_tune
from config.constants import Constants, Paths
from config.paper_experiments import get_hyperparam_tuning_config, get_training_config, get_independent_test_config, \
    get_split_strategy_config, get_common_config

torch.set_float32_matmul_precision("high")
limit_memory(Constants.MEMORY_LIMIT)

if torch.cuda.is_available():
    # print("Using GPU.")
    accelerator = "gpu"
    device = torch.device("cuda:0")
else:
    # print("Using CPU.")
    accelerator = "cpu"
    device = torch.device("cpu")


def run_experiment(config: dict):
    config = update_config(config)
    if len(find_tune_keys(config)) > 0:
        run_tune(config,
                 accelerator=accelerator,
                 datamodule_class=HDDDataModule)
        logger = get_logger(config["name"], config.get("version", None))
        logger.log_best_hyperparams(
            objective=("training/val/loss", "min"),
            versions=[f"trial-{i}" for i in range(config["ray_tune"]["num_samples"])]
        )
        logger.finalize("success")
    elif config["dataset"]["split_strategy"] == "test_only":
        if "ray_tune" in config:
            config.pop("ray_tune")
        test_func(config,
                  ds_typs=["test"]
                  )
    else:
        if "ray_tune" in config:
            config.pop("ray_tune")
        train_func(config,
                   with_tune=False
                   )


parser = argparse.ArgumentParser(description='Run experiments.')
group = parser.add_argument_group('General configuration')
group.add_argument('--models', type=str, nargs='+',
                   help='The models to run the experiments for. Possible values: MLP, LSTM, RF, HGBC', required=True)
group.add_argument('--split_strategies', type=str, nargs='+',
                   help='The split strategies to run the experiments for. Possible values: no-split, temporal, group-based, random0',
                   required=True)
group.add_argument('--random_seeds', type=int, nargs='+', help='The random seeds to run the experiments for.',
                   required=True)
group.add_argument('--test_year', type=int, help='The year to run the independent test for.', required=True)
group = parser.add_argument_group('Experiment types')
group.add_argument('--hyperparam_tuning', type=int, help='Run hyperparameter tuning experiments with specified seed.',
                   required=False)
group.add_argument('--training', action='store_true', help='Run training experiments.')
group.add_argument('--independent_test', action='store_true', help='Run independent test with specified year.')
group = parser.add_argument_group('Evaluation Procedures')
group.add_argument('--generate_tables_figures', action='store_true',
                   help='Generate remaining latex tables and figures for the paper.')
group.add_argument('--compute_leakage', action='store_true',
                   help='Compute the leakage values and write them to a csv table.')
group.add_argument('--aggregate_versions_training', action='store_true',
                   help='Aggregate the results from multiple seeds of the training runs.')
group.add_argument('--aggregate_versions_independent_test', action='store_true',
                   help='Aggregate the results from multiple seeds of the independent test runs.')

if __name__ == "__main__":

    args = parser.parse_args()

    model_names = args.models
    split_strategies = args.split_strategies

    distinct_names = set()
    for model_name in model_names:
        for split_strategy in split_strategies:
            distinct_names.add(f"model={model_name}_split={split_strategy}")

    if args.hyperparam_tuning:
        print("Running hyperparameter tuning experiments.")
        for config in get_hyperparam_tuning_config(model_names, split_strategies, args.hyperparam_tuning,
                                                   debug_mode=False):
            run_experiment(config)
        print("Finished hyperparameter tuning experiments.")
    if args.training:
        print("Running training experiments.")
        for config in get_training_config(model_names, split_strategies, args.random_seeds):
            run_experiment(config)
        print("Finished training experiments.")
    if args.training or args.aggregate_versions_training:
        print("Aggregating versions for training runs.")
        for name in distinct_names:
            aggregate_versions(name=name, versions=[f"seed-{i}" for i in args.random_seeds])
        print("Finished aggregating versions for training runs.")
    if args.independent_test:
        print("Running independent test.")
        for config in get_independent_test_config(model_names, split_strategies, args.test_year,
                                                  args.random_seeds):
            run_experiment(config)
        print("Finished independent test.")
    if args.independent_test or args.aggregate_versions_independent_test:
        print("Aggregating versions for independent test runs.")
        for name in [f"independent-test_year={args.test_year}_{n}" for n in distinct_names]:
            aggregate_versions(name=name, versions=[f"seed-{i}" for i in args.random_seeds])
        print("Finished aggregating versions for independent test runs.")
    if args.generate_tables_figures:
        print("Generating tables and figures.")
        generate_tables_and_diagrams(
            model_names=model_names,
            split_strategies=split_strategies,
            default_year="2014/2015",
            independent_test_year=str(args.test_year),
        )
        print("Finished generating tables and figures.")
    if args.compute_leakage:
        alphas = [0, 1]
        print("Computing leakage values.")
        leakage_df = defaultdict(list)
        for split_strategy in split_strategies:
            leakage_df["split_strategy"].append(split_strategy)
            common_config = get_common_config()
            dataset_config = common_config["dataset"]
            dataset_config["task"] = common_config["task"]
            dataset_config["split_strategy"] = get_split_strategy_config(split_strategy)
            datamodule = HDDDataModule(dataset_config)
            curr_leakage_values = HDDDataModule.get_leakage_value(datamodule, alphas)
            for alpha, leakage_val in curr_leakage_values.items():
                leakage_df[f"alpha={alpha}"].append(leakage_val)
        leakage_df = pd.DataFrame(leakage_df)
        Paths.TABLES.mkdir(exist_ok=True)
        leakage_df.to_csv(Paths.TABLES / "leakage_values.csv", index=False)
        print("Finished computing leakage values.")
