import datetime
import pickle
import copy
from pathlib import Path, PosixPath
from typing import Literal, Optional

import numpy as np
import torch
from torch import tensor  # Needed for the eval function
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint

from ray import train
from ray.air import session
from ray.tune.integration.pytorch_lightning import TuneReportCheckpointCallback

from config.constants import Constants
from data.data_module import HDDDataModule
from models.models import get_model, get_LitModel
from models.models_enum import PytorchModels, SKLearnModels
from training.logging import ExtendedLogger, get_logger
from training.loss_functions import get_weights, get_criterion
from utils.eval import find_thresholds, get_metrics_and_figures
from utils.misc import rec_defaultdict, flatten_dict, unflatten_dict, merge_dicts

if torch.cuda.is_available():
    # print("Using GPU.")
    accelerator = "gpu"
    device = torch.device("cuda:0")
else:
    # print("Using CPU.")
    accelerator = "cpu"
    device = torch.device("cpu")


def train_func(
        config: dict,
        with_tune: bool,
        logger: Optional[ExtendedLogger] = None,
        data_module=None,
        verbose: bool = False):
    """
    Trains a model with the given configuration. Afterward, the model is tested.
    :param config: The configuration dictionary.
    :param with_tune: Whether the function is called within a ray tune trial.
    :param logger: The logger to be used. If None, a new logger is created.
    :param data_module: The data module to be used. If None, a new data module is created.
    :param verbose: Whether to print the progress bar.
    :return: None (the results are logged to the logger; in the end, test_func is called)
    """

    pl.seed_everything(config.get("seed", Constants.DEFAULT_SEED))
    torch.set_float32_matmul_precision("high")

    if with_tune:
        trial_id = session.get_trial_id()
        trial_dirname = session.get_trial_dir()
        trial_idx = trial_dirname[trial_dirname.index(trial_id):]
        trial_idx = int(trial_idx.split("_")[1]) - 1
        config["version"] = f"trial-{trial_idx}"

    if logger is None:
        logger = get_logger(
            name=config["name"],
            version=config.get("version", None)
        )

    if data_module is None:
        config["dataset"]["log_prefix"] = config.get("version", "")
        data_module = HDDDataModule(config["dataset"], num_workers=Constants.NUM_WORKERS)
        data_module.setup("fit")

    config["model"]["input_size"] = data_module.get_input_size()

    resolve_weights(config, data_module, "training")
    criterion = get_criterion(config["training"]["loss"])

    logger.log_hyperparams(config)

    for obj_name, obj in data_module.get_objects().items():
        logger.save_object(f"saved_objects/{obj_name}", obj)

    if config["model"]["type"] in PytorchModels._member_names_:
        metrics_str_early_stopping, mode_early_stopping = config["training"]["objective"]
        base_model = get_model({**config["model"]})
        model = get_LitModel(config=config, base_model=base_model, criterion=criterion,
                             optimizer=get_optimizer_and_scheduler(base_model,
                                                                   optimizer_config=config["training"]["optimizer"],
                                                                   scheduler_config=config["training"].get(
                                                                       "lr_scheduler",
                                                                       None)))

        # callbacks = [LearningRateMonitor(logging_interval="epoch")]
        callbacks = [
            EarlyStopping(monitor=metrics_str_early_stopping,
                          mode=mode_early_stopping,
                          patience=config["training"].get("patience", 10),
                          min_delta=config["training"].get("min_delta", 0.0))]
        if with_tune:
            metrics_str_tune, _ = config["ray_tune"]["objective"]
            callbacks += [
                TuneReportCheckpointCallback(
                    metrics=[metrics_str_tune],
                    on="train_epoch_end",
                    save_checkpoints=False,
                )
            ]

        callbacks += [
            ModelCheckpoint(dirpath=logger.log_dir / "model",
                            save_top_k=1,
                            monitor=metrics_str_early_stopping,
                            mode=mode_early_stopping,
                            save_weights_only=True,
                            filename="best")
        ]

        # print(callbacks)
        trainer = pl.Trainer(max_epochs=config["training"]["max_epochs"],
                             callbacks=callbacks,
                             logger=logger,
                             # log_every_n_steps=50,
                             enable_progress_bar=verbose,
                             enable_model_summary=verbose,
                             accelerator=accelerator,
                             devices=1,
                             enable_checkpointing=True,
                             num_sanity_val_steps=0,
                             # fast_dev_run=config.get("debug", False),
                             # profiler="simple",
                             )
        t0 = datetime.datetime.now()
        trainer.fit(model=model,
                    datamodule=data_module
                    )
        duration = datetime.datetime.now() - t0
        logger.save_string("training/duration", str(duration))

        logger.log_model_summary(model=model, max_depth=-1)
        # trainer.test(ckpt_path="best", datamodule=data_module, verbose=False)

    elif config["model"]["type"] in SKLearnModels._member_names_:
        trainer = None
        if isinstance(config["task"], list) or isinstance(config["task"], tuple):
            assert len(config["task"]) == 1, "Only one task is supported for sklearn models."
            task = config["task"][0]
        else:
            task = config["task"]
        assert task in ["binary", "multiclass"], "Only binary and multiclass tasks are supported for sklearn models."

        X_train = data_module.X_train
        y_train = data_module.y_train[0]

        model = get_model({**config["model"]})
        t0 = datetime.datetime.now()
        model.fit(X_train, y_train)
        duration = datetime.datetime.now() - t0
        logger.save_string("training/duration", str(duration))

        # save the trained model
        logger.save_object("model/best", model)
        config["fitted_model"] = model

        X_val = data_module.X_val
        y_val = data_module.y_val[0]
        y_hat = model.predict_proba(X_val)
        val_loss = criterion(
            torch.tensor(y_hat, device=device, dtype=torch.float32),
            torch.tensor(y_val, device=device).long()).mean().detach().cpu().item()
        logger.log_metrics({"training/val/loss": val_loss})
        if with_tune:
            tune_report_dict = dict()
            # Calculate the loss and report it to tune
            tune_report_dict["val/loss"] = val_loss
            train.report(tune_report_dict)

    else:
        raise ValueError(f"Model {config['model']['name']} is not supported.")

    test_func(config=config,
              trainer=trainer,
              logger=logger,
              data_module=data_module,
              verbose=verbose,
              ds_typs=["train", "val", "test"])


def test_func(config: dict,
              logger: Optional[ExtendedLogger] = None,
              trainer: Optional[pl.Trainer] = None,
              data_module: Optional[HDDDataModule] = None,
              verbose: bool = False,
              ds_typs: Optional[str | list[str]] = None) -> None:
    """
    Tests a model with the given configuration.
    :param config: The configuration dictionary.
    :param logger: The logger object (instance of ExtendedLogger) to be used for logging the metrics. If None,
    :param trainer: The pl trainer to be used for testing the model. If None, a new trainer is created.
    :param data_module: The data module to be used. If None, a new data module is created according to the config.
    :param verbose: Whether to print the progress bar.
    :param ds_typs: The dataset types (train/test/val) to be used for testing. If None, all three types are used.
    :return: None (the results are logged to the logger, logger will be finalized and object deleted in the end)
    """

    if not isinstance(config["task"], list):
        config["task"] = [config["task"]]
    if "multiclass" in config["task"]:
        if "pos_classes" not in config["training"].keys():
            raise ValueError(
                "The key 'pos_classes' must be present in the training config for multiclass classification task.")

    if ds_typs is None:
        ds_typs = ["train", "val", "test"]
    elif isinstance(ds_typs, str):
        ds_typs = [ds_typs]
    if logger is None:
        logger = get_logger(
            name=config["name"],
            version=config.get("version", None)
        )

    if data_module is None:
        data_module = HDDDataModule(config["dataset"])
        data_module.setup()

    if config["model"]["type"] in PytorchModels._member_names_:
        dataloaders = []
        if "train" in ds_typs:
            dataloaders.append(data_module.train_dataloader(shuffle=False))
        if "val" in ds_typs:
            dataloaders.append(data_module.val_dataloader())
        if "test" in ds_typs:
            dataloaders.append(data_module.test_dataloader())
        if trainer is None:
            trainer = pl.Trainer(max_epochs=config["training"]["max_epochs"],
                                 callbacks=None,
                                 logger=logger,
                                 # log_every_n_steps=50,
                                 enable_progress_bar=verbose,
                                 enable_model_summary=verbose,
                                 accelerator=accelerator,
                                 devices=1,
                                 enable_checkpointing=True,
                                 num_sanity_val_steps=0,
                                 # fast_dev_run=config.get("debug", False),
                                 # profiler="simple",
                                 )
            resolve_weights(config, data_module, "inference")
            criterion = get_criterion(config["training"]["loss"], scaler_y=data_module.scaler_y)
            base_model = get_model({**config["model"]})
            model = get_LitModel(base_model=base_model,
                                 config=config,
                                 criterion=criterion,
                                 optimizer=get_optimizer_and_scheduler(base_model,
                                                                       optimizer_config=config["training"]["optimizer"],
                                                                       scheduler_config=config["training"].get(
                                                                           "lr_scheduler",
                                                                           None)),
                                 ckpt_path=config.pop("fitted_model"))
            all_preds_and_true = trainer.predict(model=model,
                                                 dataloaders=dataloaders,
                                                 return_predictions=True)
        else:
            all_preds_and_true = trainer.predict(ckpt_path="best", dataloaders=dataloaders, return_predictions=True)

        if len(dataloaders) == 1:
            all_preds_and_true = [all_preds_and_true]

        # all_preds_and_true has the following format:
        # list of length 3 (for train, val and test)
        # | (each element)
        # -> list of length = no. batches
        #    | (each element)
        #    -> dictionary with {"predictions": {task_0: np.ndarray [N_batch x C0], task_1: np.ndarray [N_batch x C1], ..., task_n: np.ndarray [N_batch x Cn]} , "true": <similar> }

        all_preds_and_true_concatenated = rec_defaultdict()
        # concatenate the prediction batches for train/val/test and each task
        for preds_and_true_typ_i, typ in zip(all_preds_and_true, ds_typs):
            for task_i in (config["task"] if isinstance(config["task"], list) else [config["task"]]):
                _preds = np.concatenate([d["predictions"][task_i] for d in preds_and_true_typ_i], axis=0)
                _true = np.concatenate([d["true"][task_i] for d in preds_and_true_typ_i], axis=0)
                all_preds_and_true_concatenated[typ][task_i] = {"predictions": _preds, "true": _true}
        all_preds_and_true = all_preds_and_true_concatenated

        # Now it has the simpler format:
        # {"train": {task_0: {"predictions": np.ndarray [N x C0], "true": np.ndarray [N]},
        #            task_1: {<similar>},
        #            ...,
        #            task_n: {<similar>}}
        #  "val":   {<same>},
        #  "test":  {<same>}
        #  }
    elif config["model"]["type"] in SKLearnModels._member_names_:
        if isinstance(config["task"], list) or isinstance(config["task"], tuple):
            assert len(config["task"]) == 1, "Only one task is supported for sklearn models."
            task = config["task"][0]
        else:
            task = config["task"]
        assert task in ["binary", "multiclass"], "Only binary and multiclass tasks are supported for sklearn models."

        X = dict()
        y = dict()
        for typ in ds_typs:
            X[typ] = getattr(data_module, f"X_{typ}")
            y[typ] = getattr(data_module, f"y_{typ}")[0]

        # Get fitted model
        model = config.pop("fitted_model")
        if isinstance(model, str):
            model = Path(model)
        if (isinstance(model, Path) or isinstance(model, PosixPath)) and model.suffix in [".pkl", ".ckpt"]:
            with open(model, "rb") as f:
                model = pickle.load(f)
        if hasattr(model, "predict_proba"):
            # Model is loaded and has the predict_proba method
            pass
        else:
            raise ValueError(
                f"Invalid fitted_model input. It should be a file path (str or Path) to a .pkl file or a trained model object with a 'predict_proba' method. Instead, got: {model} of type {type(model)}")

        all_preds_and_true = rec_defaultdict()
        for typ in ds_typs:
            curr_y = y[typ]
            curr_X = X[typ]

            y_hat = model.predict_proba(curr_X)

            all_preds_and_true[typ][task] = {"predictions": y_hat, "true": curr_y}

    else:
        raise ValueError(f"Model {config['model']['name']} is not supported.")

    # Save the predictions and true values:
    for path, np_array in flatten_dict(all_preds_and_true).items():
        logger.save_object(f"{Constants.LOGGING_PREDICTIONS_PREFIX}/{path}", np_array)

    # If optimal thresholds are passed with the config, these are used
    # Otherwise, optimal thresholds are found based on the validation predictions and true values and
    # with respect to the threshold_metric specified in the config
    if "optimal_thresholds" in config:
        optimal_thresholds = config["optimal_thresholds"]
    elif "threshold_metric" in config["training"].keys() and isinstance(config["training"]["threshold_metric"], tuple):
        optimal_thresholds = find_thresholds(
            all_preds_and_true=all_preds_and_true,
            threshold_metrics=config["training"]["threshold_metric"],
            pos_classes=config["training"].get("pos_classes", None),
        )
    else:
        optimal_thresholds = None
    if optimal_thresholds is not None:
        logger.log_metrics({f"{Constants.LOGGING_THRESHOLD_PREFIX}/{k}": v for k, v in
                            flatten_dict(optimal_thresholds).items()})

    metrics = get_metrics_and_figures(all_preds_and_true=all_preds_and_true,
                                      thresholds=unflatten_dict(
                                          optimal_thresholds) if optimal_thresholds is not None else None,
                                      pos_classes=config["training"]["pos_classes"],
                                      figure_consumer=lambda fig, path: logger.log_figure(path, fig))

    logger.log_metrics(metrics)
    logger.finalize("success")
    del logger


def update_config(config: dict) -> dict:
    """
    Updates the configuration dictionary with the necessary information for training and testing.
    The function works on a deep copy of the configuration dictionary. The original dictionary is not modified.
    :param config: The configuration dictionary.
    :return: The updated configuration dictionary.
    """
    config = copy.deepcopy(config)
    if "load_from" in config:
        if isinstance(config["load_from"], str):
            config["load_from"] = {
                "name": config["load_from"],
                "version": None,
            }
        logger = get_logger(config["load_from"]["name"], version=config["load_from"]["version"])
        assert logger.fetch_string(
            "status") == "success", f"Experiment {config['load_from']['name']} must have status 'success' to get saved objects."
        fetched_config = logger.get_hyperparams()
        del logger
        config = merge_dicts(config, fetched_config)

    if "train_hdd_models" in config["dataset"].keys():
        config["dataset"]["hdd_models"] = config["dataset"]["train_hdd_models"]

    if config["model"]["type"] in PytorchModels._member_names_:
        config["model"]["task"] = config["task"]
        if torch.cuda.is_available():
            accelerator = "gpu"
            device = torch.device("cuda:0")
        else:
            accelerator = "cpu"
            device = torch.device("cpu")
    else:
        accelerator = "cpu"
        device = torch.device("cpu")
    config["hardware"] = {"accelerator": accelerator, "device": device}

    if not isinstance(config["dataset"], str):
        config["dataset"]["seed"] = config.get("seed", Constants.DEFAULT_SEED)
        config["dataset"]["task"] = config["task"]
        if config["model"]["type"] in PytorchModels._member_names_:
            config["dataset"]["batch_size"] = config["training"]["batch_size"]

    if "load_objects" in config["dataset"].keys():
        if "name" not in config["dataset"]["load_objects"].get("from", {}):
            raise ValueError("Name of the logger to load objects from must be provided.")
        if "version" not in config["dataset"]["load_objects"].get("from", {}):
            config["dataset"]["load_objects"]["from"]["version"] = None
        requested_object_names = config["dataset"]["load_objects"]["names"]
        if isinstance(requested_object_names, str):
            requested_object_names = [requested_object_names]
        logger = get_logger(config["dataset"]["load_objects"]["from"]["name"],
                            version=config["dataset"]["load_objects"]["from"]["version"])
        config["dataset"]["loaded_objects"] = dict()
        for object_name in requested_object_names:
            obj = logger.fetch_object(f"saved_objects/{object_name}")
            if obj is None:
                raise ValueError(
                    f"Object {object_name} not found in logger {logger.name} with version={logger.version}.")
            config["dataset"]["loaded_objects"][object_name] = obj
        del logger

    return config


def resolve_weights(config: dict, data_module: HDDDataModule, mode: Literal["training", "inference"]) -> None:
    """
    Resolves the weights for the loss function in the configuration dictionary. In other words:
    Replaces the loss weighting strategy string (if any) by the actual weights for the specific dataset.
    Alters the configuration dictionary in place.
    :param config: The configuration dictionary.
    :param data_module: The data module to be used.
    :param mode: The mode to be used. Either "training" or "inference".
    """
    if isinstance(config["training"]["loss"], list):
        for i, _loss_config_i in enumerate(config["training"]["loss"]):
            if "params" in _loss_config_i and "weight" in _loss_config_i["params"]:
                weight_param = _loss_config_i["params"].get("weight", None)
                if isinstance(weight_param, str):
                    if mode == "inference":
                        config["training"]["loss"][i]["params"]["weight"] = get_weights(
                            "equal",
                            data_module.y_counts["class"]["test"]["absolute"],
                            device=device
                        )
                    elif weight_param.startswith("tensor("):
                        config["training"]["loss"][i]["params"]["weight"] = eval(weight_param).to(device)
                    else:
                        config["training"]["loss"][i]["params"]["weight"] = get_weights(
                            weight_param,
                            data_module.y_counts["class"]["train"]["absolute"],
                            device=device
                        )
                elif isinstance(weight_param, list):
                    config["training"]["loss"][i]["params"]["weight"] = torch.tensor(weight_param, device=device)
                else:
                    raise ValueError(f"Invalid loss weight parameter {weight_param} in loss config.")
    else:
        if "params" in config["training"]["loss"] and "weight" in config["training"]["loss"]["params"]:
            weight_param = config["training"]["loss"]["params"].get("weight", None)
            if isinstance(weight_param, str):
                if mode == "inference":
                    config["training"]["loss"]["params"]["weight"] = get_weights(
                        "equal",
                        data_module.y_counts["class"]["test"]["absolute"],
                        device=device
                    )
                elif weight_param.startswith("tensor("):
                    config["training"]["loss"]["params"]["weight"] = eval(weight_param).to(device)
                else:
                    config["training"]["loss"]["params"]["weight"] = get_weights(
                        weight_param,
                        data_module.y_counts["class"]["train"]["absolute"],
                        device=device
                    )
            elif isinstance(weight_param, list):
                config["training"]["loss"]["params"]["weight"] = torch.tensor(weight_param, device=device)
            elif isinstance(weight_param, torch.Tensor):
                pass
            else:
                raise ValueError(f"Invalid loss weight parameter {weight_param} in loss config.")


def get_optimizer_and_scheduler(model: torch.nn.Module, optimizer_config: dict, scheduler_config: dict = None):
    """
    In the config, there are the keys "optimizer", "lr" and "lr_scheduler" with string values. Returns the optimizer and the scheduler for pytorch lightning. "lr_scheduler" may also be None or the key may not be present.
    :param optimizer_config: The config dictionary for the optimizer, containing at least the name
    :param scheduler_config: The config dict for the scheduler
    :param model: The model to be optimized.
    :return: dict: optimizer and the lr_scheduler.
    """
    opt_params = optimizer_config.get("params", {})
    if optimizer_config["name"] == "Adam":
        optimizer = torch.optim.Adam(model.parameters(), **opt_params)
    elif optimizer_config["name"] == "AdamW":
        optimizer = torch.optim.AdamW(model.parameters(), **opt_params)
    elif optimizer_config["name"] == "RMSprop":
        optimizer = torch.optim.RMSprop(model.parameters(), **opt_params)
    elif optimizer_config["name"] == "SGD":
        optimizer = torch.optim.SGD(model.parameters(), **opt_params)
    else:
        raise ValueError("Optimizer must be 'Adam', 'SGD', 'AdamW' or 'RMSprop'.")

    if scheduler_config is None or len(scheduler_config) == 0:
        return optimizer

    scheduler_params = scheduler_config.get("params", {})
    if scheduler_config["name"] == "ReduceLROnPlateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, **scheduler_params)
    elif scheduler_config["name"] == "StepLR":
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, **scheduler_params)
    elif scheduler_config["name"] == "CosineAnnealingLR":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, **scheduler_params)
    elif scheduler_config["name"] == "CosineAnnealingWarmRestarts":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, **scheduler_params)
    else:
        raise ValueError(
            "lr_scheduler must be 'ReduceLROnPlateau', 'StepLR', 'CosineAnnealingLR' or 'CosineAnnealingWarmRestarts'.")

    return {"optimizer": optimizer,
            "lr_scheduler": scheduler}
