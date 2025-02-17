import copy
from typing import Optional

from ray import tune

from config.constants import Constants
from training.logging import get_logger


def get_common_config() -> dict:
    """
    Get the common configuration dict for all experiments.
    :return: The common configuration dict.
    """
    return {
        "seed": 0,
        "task": "multiclass",
        "dataset": {
            "scaler": {"StandardScaler": "all_normalized", "QuantileTransformer": "all_raw"},
            "scaler_y": "StandardScaler",
            "time_periods": [2014, 2015],
            "class_intervals": [7, 28],
            "additional_inf_class": True,
            "inf_sample_size": 0.025,
            "lags": list(range(1, 8)),
            "hdd_models": Constants.SEAGATE_4TB,
            "max_rul_threshold": None,
            "rul_cap_value": None,
            "T_gap": 3,
            "features": "paper",
            "replace_smart_random": False,  # This is only for debugging and should be set to False for proper training
        },
        "training": {
            "pos_classes": [0],
            "threshold_metric": ("MCC", "max"),
            "loss":
                {
                    "name": "CE", "params": {"weight": "balanced"}
                },
            "objective": ("val/loss", "min"),
            "optimizer": {"name": "AdamW",
                          "params": {
                              "lr": tune.loguniform(1e-6, 1e-3),  # 1e-6
                              "weight_decay": 0}},
            "batch_size": 32,
            "max_epochs": 1000,
            "patience": 15,
            "min_delta": 0.01,
        },
        "ray_tune": {
            "scheduler": {
                "name": "MedianStoppingRule",
            },
            "search_alg": {
                "name": "HyperOpt",
            },
            "max_concurrent_trials": 10,
            "total_cpus": 10,
            "restore": False,
        }
    }


def get_split_strategy_config(split_strategy_name: str) -> dict:
    """
    Definition of the different split strategies for splitting the dataset into training, validation, and test sets.
    :param split_strategy_name: The name of the split strategy.
    :return: Dictionary of split strategy configurations.
    """
    if split_strategy_name == "no-split":
        return {"no_split": True}
    elif split_strategy_name == "group-based":
        return {"overlap_HDD_ratio": .0}
    elif split_strategy_name == "random0":
        return {
            "overlap_HDD_ratio": 1.,
            "temporal_split_ratio": .0,
        }
    elif split_strategy_name == "temporal":
        return {
            "temporal_split_dates": ("2015-03-14", "2015-05-25"),
            "temporal_split_ratio": 1.,
            "overlap_HDD_ratio": 1.,
        }
    else:
        raise ValueError(f"Unknown split strategy: {split_strategy_name}")


def get_model_config(model_name: str) -> dict:
    """
    The model configurations and hyperparameter spaces for the hyperparameter tuning.
    :param model_name: The name of the model.
    :return: Dictionary of model configurations.
    """
    if model_name == "LSTM":
        return {
            "type": "LSTM",
            "input_size": "auto",
            "hidden_size": tune.qrandint(64, 256, 32),
            "output_size": 4,
            "bidirectional": False,
            "num_layers": tune.randint(1, 3),
            "bias": True,
        }
    elif model_name == "MLP":
        return {
            "type": "MLP",
            "input_size": "auto",
            "hidden_sizes": [tune.qrandint(32, 256, 32) for _ in range(3)],  # => +30 trials
            "output_size": 4,
            "batch_norm": False,
            "activation": "relu",
        }
    elif model_name == "RF":
        return {"type": "RF",
                "n_estimators": tune.randint(50, 500),
                "max_depth": tune.randint(5, 100),
                "max_features": tune.uniform(0.1, 1.0),
                "bootstrap": True,
                "n_jobs": 30,
                }
    elif model_name == "HGBC":
        return {"type": "HGBC",
                "learning_rate": tune.uniform(0.01, 1.0),
                "max_iter": tune.randint(50, 500),
                "max_leaf_nodes": tune.randint(5, 100),
                "max_features": tune.uniform(0.1, 1.0),
                "class_weight": tune.choice(["balanced", None]), }
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_num_trials(config: dict) -> int:
    """
    Helper function to get the number of trials for the hyperparameter tuning.
    The numbers are chosen according to the guidance by Sean Owen:
    https://www.databricks.com/blog/2021/04/15/how-not-to-tune-your-model-with-hyperopt.html
    April 15, 2021
    """
    model_name = config["model"]["type"]
    if model_name == "HGBC":
        return 70
    elif model_name == "MLP":
        return 40
    elif model_name == "LSTM":
        return 30
    elif model_name == "RF":
        return 30
    else:
        raise ValueError(f"Unspecified num_trials for model type: {model_name}")


def get_hyperparam_tuning_config(
        models: str | list[str],
        split_strategies: str | list[str],
        random_seed: int,
        debug_mode: bool = False,
) -> list[dict]:
    """
    Returns a list of config dictionaries for the hyperparameter tuning of the specified models and split strategies.
    :param models: Model(s) to run the hyperparameter tuning for.
    :param split_strategies: Split strategies to run the hyperparameter tuning for.
    :param random_seed: The random seed used for hyperparamter tuning.
    :param debug_mode: If set to true, the number of epochs and trials is set to 2, respectively.
    :return: List of configuration dicts. Length: #models x #split_strategies
    """
    res = []
    for model_name in models:
        model_conf_dict = get_model_config(model_name)
        for split_strategy_name in split_strategies:
            curr_exp_config: dict = get_common_config()

            if model_name not in ["LSTM", "MLP"]:
                # Only keep hyperparameters that are relevant for the model
                curr_exp_config["training"] = {k: v for k, v in curr_exp_config["training"].items() if
                                               k in ["pos_classes", "threshold_metric", "loss", "objective"]}

            curr_name = f"hyperparam-search_model={model_name}_split={split_strategy_name}"

            curr_exp_config["debug"] = debug_mode
            if debug_mode:
                curr_exp_config["training"]["max_epochs"] = 2

            curr_exp_config["name"] = curr_name
            curr_exp_config["seed"] = random_seed

            curr_exp_config["model"] = copy.deepcopy(model_conf_dict)
            curr_exp_config["dataset"]["split_strategy"] = get_split_strategy_config(split_strategy_name)

            if "ray_tune" in curr_exp_config.keys():
                curr_exp_config["ray_tune"][
                    "num_samples"] = get_num_trials(curr_exp_config) if not debug_mode else 2
            if model_name == "LSTM":
                curr_exp_config["dataset"]["sequential"] = True
            res.append(curr_exp_config)
    return res


def get_training_config(
        models: str | list[str],
        split_strategies: str | list[str],
        random_seeds: int | list[int],
        config: Optional[dict] = None
) -> list[dict]:
    """
    Returns a list of config dictionaries for the training of the models for different split strategies.
    The experiments are repeated #random_seeds times. Assumes, that for each pair of model and split_strategy,
    hyperparameter tuning has been conducted before.
    :param models: Model(s) to run the experiments for.
    :param split_strategies: Split strategies to run the experiments for.
    :param random_seeds: Random seed(s) for the experiments.
    :param config: Optional configuration dict to overwrite standard behavior, that is, picking the best hyperparameters from the hyperparameter tuning.
    :return: List of configuration dicts. Length: #models x #split_strategies x #random_seeds
    """
    res = []
    if isinstance(random_seeds, int):
        random_seeds = [random_seeds]
    # For the best hyperparameters, repeat training with multiple random seeds and evaluate the model on the test sets
    for model_name in models:
        for split_strategy_name in split_strategies:
            if config is None:
                hyperparam_search_exp_name = f"hyperparam-search_model={model_name}_split={split_strategy_name}"
                logger = get_logger(hyperparam_search_exp_name)
                best_config = logger.get_best_hyperparams()
                best_config["name"] = hyperparam_search_exp_name.replace("hyperparam-search_", "")
            else:
                best_config = copy.deepcopy(config)
                best_config["name"] = f"model={model_name}_split={split_strategy_name}"
            best_config.pop("ray_tune", None)
            best_config["dataset"]["save_objects"] = ["scaler"]  # save the scaler for the independent test
            for seed in random_seeds:
                curr_config = copy.deepcopy(best_config)
                curr_config["seed"] = seed
                curr_config["version"] = f"seed-{seed}"
                res.append(curr_config)
    return res


def get_independent_test_config(
        models: str | list[str],
        split_strategies: str | list[str],
        year: int,
        random_seeds: int | list[int],
):
    """
    Returns a list of config dictionaries for testing trained models on the HDD data from <year>.
    Assumes, that for each combination of (model, split_strategy, random_seed) a training as been run before and the
    trained model has been saved.
    :param models: Model(s) to run the experiments for.
    :param split_strategies: Split strategies to run the experiments for.
    :para year: The year of the Backblaze HDD data to test the models on.
    :param random_seeds: Random seed(s) for the experiments.
    :return: List of configuration dicts. Length: #models x #split_strategies x #random_seeds
    """
    res = []
    time_periods = [year]
    for model_name in models:
        for split_strategy_name in split_strategies:
            original_training_name = f"model={model_name}_split={split_strategy_name}"
            for seed in random_seeds:
                # get logger of the original training
                original_training_logger = get_logger(original_training_name, f"seed-{seed}")

                # get the hyperparameters of the original training
                curr_config = original_training_logger.get_hyperparams()

                # get the path where the fitted model is saved

                fitted_model_path = original_training_logger.fetch_object("model/best", return_obj_path=True)
                curr_config["fitted_model"] = fitted_model_path

                # make config adaptations for the independent test
                curr_config["name"] = f"independent-test_year={year}_{original_training_name}"
                curr_config["dataset"]["time_periods"] = time_periods
                curr_config["dataset"]["split_strategy"] = "test_only"
                curr_config["dataset"]["load_objects"] = {"from": {"name": original_training_name,
                                                                   "version": f"seed-{seed}"},
                                                          "names": ["scaler"]}  # load scaler of the original training

                # get thresholds from training run
                optimal_thresholds = original_training_logger.fetch_all_values("thresholds")
                curr_config["optimal_thresholds"] = optimal_thresholds

                res.append(curr_config)
    return res
