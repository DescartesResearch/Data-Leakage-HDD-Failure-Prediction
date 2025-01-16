import fnmatch
from collections import defaultdict
from pathlib import Path
from typing import Optional
from lightning.pytorch import LightningModule
import torch
import torch.nn as nn
import numpy as np
from lightning.pytorch.loggers import NeptuneLogger
from torch.utils.data import TensorDataset

from config.constants import Constants
from models.models_enum import PytorchModels, SKLearnModels
from utils.eval import get_metrics_and_figures

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_activation_function(activation: str) -> nn.modules.activation:
    """
    Returns the activation function based on the given string.
    :param activation: The name of the activation function.
    :return: The activation function.
    """
    if activation == "relu":
        return nn.ReLU()
    elif activation == "sigmoid":
        return nn.Sigmoid()
    elif activation == "tanh":
        return nn.Tanh()
    elif activation == "leaky_relu":
        return nn.LeakyReLU()
    elif activation == "gelu":
        return nn.GELU()
    else:
        raise NotImplementedError(f"Activation function {activation} is not implemented.")


class MLP(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_sizes: list[int],
                 output_size: int | list[int],
                 batch_norm: bool,
                 activation_function: str,
                 dropout_input: float,
                 dropout_hidden: float):
        """
        Initialize the MLP model.
        :param input_size: Number of features of the input.
        :param hidden_sizes: A list of hidden layer sizes. Results in len(hidden_sizes) hidden layers with the specified sizes.
        :param output_size: Number of features of the output. If a list of integers is provided, the model will output a tuple of multiple values. For each output, a seperate linear layer is used.
        :param batch_norm: If True, batch normalization is used after each linear layer (except the output layer(s)).
        :param activation_function: The activation function to use.
        :param dropout_input: The dropout rate for the input layer.
        :param dropout_hidden: The dropout rate for the hidden layers.
        """
        super(MLP, self).__init__()
        self.input = nn.Linear(input_size, hidden_sizes[0])
        self.hidden = nn.Sequential()
        if batch_norm:
            self.hidden.add_module("bn_input", nn.BatchNorm1d(hidden_sizes[0]))
        self.hidden.add_module("act_input", get_activation_function(activation_function))
        self.hidden.add_module("dropout_input", nn.Dropout(dropout_input))
        for i in range(1, len(hidden_sizes)):
            self.hidden.add_module(f"fc{i}", nn.Linear(hidden_sizes[i - 1], hidden_sizes[i]))
            if batch_norm:
                self.hidden.add_module(f"bn{i}", nn.BatchNorm1d(hidden_sizes[i]))
            self.hidden.add_module(f"act{i}", get_activation_function(activation_function))
            self.hidden.add_module(f"dropout{i}", nn.Dropout(dropout_hidden))
        # self.network.add_module("output", nn.Linear(hidden_sizes[-1], output_size))
        self.proj = nn.ModuleList()
        output_size = [output_size] if isinstance(output_size, int) else output_size
        for i, output_size_i in enumerate(output_size):
            self.proj.append(nn.Linear(hidden_sizes[-1], output_size_i))

    def forward(self, x) -> tuple[torch.Tensor] | torch.Tensor:
        """
        Forward pass of the MLP model.
        :param x: The input tensor of shape (*, input_size).
        :return: The output tensor of shape (*, output_size) or a tuple of output tensors of shape (*, output_size_i) for i in range(len(output_size)).
        """
        x = self.input(x)
        x = self.hidden(x)
        if len(self.proj) > 1:  # multi-output case
            y = tuple(linear(x) for linear in self.proj)
        else:
            y = self.proj[0](x)
        return y


class LSTM(nn.Module):
    def __init__(self,
                 input_size: int,
                 hidden_size: int,
                 num_layers: int,
                 bias: bool,
                 dropout: float,
                 bidirectional: bool,
                 proj_size: int | list[int] | None):
        """
        Initialize the LSTM model.
        :param input_size: Number of features of the input.
        :param hidden_size: Number of dimensions for the hidden states.
        :param num_layers: Number of stacked LSTM layers.
        :param bias: If False, no bias weights are used.
        :param dropout: If non-zero, use dropout between the LSTM layers.
        :param bidirectional: If True, becomes a bidirectional LSTM, else unidirectional.
        :param proj_size: The output size(s) of the final linear layer(s). If None, the hidden states h_n, c_n will be returned.
        """
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,
                            num_layers=num_layers,
                            bias=bias,
                            batch_first=True,
                            dropout=dropout,
                            bidirectional=bidirectional)
        self.bidirectional = bidirectional
        if proj_size is not None:
            self.proj = nn.ModuleList()
            proj_size = [proj_size] if isinstance(proj_size, int) else proj_size
            for i, output_size_i in enumerate(proj_size):
                self.proj.append(nn.Linear(hidden_size * (2 if bidirectional else 1), output_size_i))
        else:
            self.proj = None

    def forward(self, x) -> tuple[torch.Tensor] | torch.Tensor | tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass of the LSTM model.
        D = 2 if bidirectional else 1
        Final hidden state h_n is of shape (D*num_layers, batch_size, hidden_size)
        Final hidden state c_n is of shape (D*num_layers, batch_size, hidden_size)
        output is of shape (batch_size, seq_len, D*hidden_size) if self.proj is not None
        else: returns output, (h_n, c_n)
        :param x: The input tensor of shape (batch_size, seq_len, input_size).
        :return: The output tensor as described above. For the multi-output case (proj_size is a list), a tuple of output tensors from multiple output linear layers is returned.
        """

        output, (h_n, c_n) = self.lstm(x)

        if self.proj is None:
            return output, (h_n, c_n)

        # c_n = self._get_last_layer_hidden_state(hn_or_cn=c_n, batch_size=x.size(0))
        h_n = self._get_last_layer_hidden_state(hn_or_cn=h_n, batch_size=x.size(0))
        if len(self.proj) > 1:  # multi-output case
            y = tuple(
                linear(h_n) for linear in
                self.proj)  # shape: ((batch_size, proj_size_i) for i in range(len(output_size))
        else:
            y = self.proj[0](h_n)

        return y

    def _get_last_layer_hidden_state(self, hn_or_cn: torch.Tensor, batch_size):
        """
        In the unidirectional lstm, the last hidden state can simply be accessed by index -1.
        However, in the bidirectional case, we must concatenate the states from the forward and backward lstm.
        :param hn_or_cn: hn or cn of shape (D*num_layers, batch_size, hidden_size)
        :return: hn or cn of shape (batch_size, 2*hidden_size)
        """
        if self.bidirectional:
            hn_or_cn = hn_or_cn.view(self.lstm.num_layers, 2, batch_size,
                                     self.lstm.hidden_size)  # shape: (2, num_layers, batch_size, hidden_size)
            hn_or_cn = torch.cat((hn_or_cn[-1, 0], hn_or_cn[-1, 1]), dim=1)  # shape: (batch_size, 2*hidden_size)
            # _tmp = output.view(output.shape[0], output.shape[1], 2, self.lstm.hidden_size)  # shape: (batch_size, L, 2, hidden_size)
            # _tmp = torch.cat((_tmp[:, -1, 0], _tmp[:, 0, 1]), dim=1) # shape: (batch_size, 2*hidden_size)
            # assert ((_tmp - h_n) < 1e-6).all() # already tested, for efficiency we skip this
        else:
            hn_or_cn = hn_or_cn[-1]
            # assert ((h_n - output[:, -1, :]) < 1e-6).all() # already tested, for efficiency we skip this
        return hn_or_cn


class LitModel(LightningModule):
    """
    The LightningModule for training and evaluating the Deep Learning models with PyTorch Lightning.
    """

    def __init__(self,
                 model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 task: str | list[str],
                 pos_classes: list = None):
        """
        Initialize the LightningModule.
        :param model: The PyTorch model to train.
        :param criterion: The loss function to use.
        :param optimizer: The optimizer to use.
        :param task: The task(s) to perform.
        :param pos_classes: The positive classes for the "multiclass" classification task.
        """
        super(LitModel, self).__init__()
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.task = [task] if isinstance(task, str) else task
        self.pos_classes = pos_classes
        self.epoch_outputs = {
            _typ: {_task: {"predictions": [], "true": []} for _task in self.task} for _typ in ["train", "val", "test"]
        }
        self.curr_epoch_losses = defaultdict(lambda: defaultdict(list))
        self.log_prefix = f"training/"

    def forward(self, x):
        return self.model(x)

    def _step(self, typ, batch, batch_idx):
        x, *y_list = batch
        y = y_list[0] if len(y_list) == 1 else tuple(y_list)
        # For the multi-output case, y is a tuple of tensors and loss is a list of losses of the same length
        y_pred = self.forward(x)
        loss = self.criterion(y_pred, y)

        if isinstance(loss, list):
            # Get the losses for each task
            for loss_i, task_i in zip(loss, self.task):
                self.curr_epoch_losses[typ][f"{task_i}/loss"].append(loss_i.mean())
                self.log(f"{typ}/{task_i}/loss", loss_i.mean(), on_step=False, on_epoch=True, logger=False)
                # self.log(f"{typ}/batch/loss_{i}", loss_i, on_step=True, on_epoch=False)
            loss = sum(loss)
        # Get the total loss
        loss = torch.mean(loss)
        self.curr_epoch_losses[typ]["loss"].append(loss)
        self.log(f"{typ}/loss", loss, on_step=False, on_epoch=True, logger=False)

        # Save the predictions and true values for the current epoch for evaluation at the end of the epoch
        if not isinstance(y, tuple) and not isinstance(y, list):
            y = (y,)
        if not isinstance(y_pred, tuple) and not isinstance(y_pred, list):
            y_pred = (y_pred,)
        for y_pred_i, y_i, task_i in zip(y_pred, y, self.task):
            y_pred_i_np = y_pred_i.detach().cpu().numpy()
            y_i_np = y_i.detach().cpu().numpy()

            self.epoch_outputs[typ][task_i]["predictions"].append(y_pred_i_np)
            self.epoch_outputs[typ][task_i]["true"].append(y_i_np)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step("train", batch, batch_idx)

    def validation_step(self, batch, batch_idx):
        return self._step("val", batch, batch_idx)

    def test_step(self, batch, batch_idx):
        return self._step("test", batch, batch_idx)

    def predict_step(self, batch, batch_idx, dataloader_idx=0) -> dict:
        x, *y_list = batch
        y = y_list[0] if len(y_list) == 1 else tuple(y_list)
        y_pred = self.model(x)

        if not isinstance(y, tuple):
            y = (y,)
        if not isinstance(y_pred, tuple):
            y_pred = (y_pred,)

        pred_outputs = {
            "predictions": {},
            "true": {},
        }
        for y_pred_i, y_i, task_i in zip(y_pred, y, self.task):
            y_pred_i_np = y_pred_i.detach().cpu().numpy()
            y_i_np = y_i.detach().cpu().numpy()

            if task_i == "regression":
                y_pred_i_np = np.clip(y_pred_i_np, 0, None).reshape(-1, 1)
                y_i_np = np.clip(y_i_np, 0, None).reshape(-1, 1)
                if hasattr(self.trainer.datamodule, "scaler_y"):
                    y_pred_i_np = self.trainer.datamodule.scaler_y.inverse_transform(y_pred_i_np)
                    y_i_np = self.trainer.datamodule.scaler_y.inverse_transform(y_i_np)

            pred_outputs["predictions"][task_i] = y_pred_i_np
            pred_outputs["true"][task_i] = y_i_np

        return pred_outputs

    def _on_epoch_end(self, typ):
        # Calculate metrics for logging

        for task_i in self.task:
            # Concatenate the predictions and true values for the current epoch
            self.epoch_outputs[typ][task_i]["true"] = np.concatenate(
                self.epoch_outputs[typ][task_i]["true"], axis=0)
            self.epoch_outputs[typ][task_i]["predictions"] = np.concatenate(
                self.epoch_outputs[typ][task_i]["predictions"], axis=0)

            if task_i == "regression":
                self.epoch_outputs[typ][task_i]["true"] = np.clip(self.epoch_outputs[typ][task_i]["true"], 0,
                                                                  None).reshape(-1, 1)
                self.epoch_outputs[typ][task_i]["predictions"] = np.clip(self.epoch_outputs[typ][task_i]["predictions"],
                                                                         0, None).reshape(-1, 1)
                if hasattr(self.trainer.datamodule, "scaler_y"):
                    self.epoch_outputs[typ][task_i]["true"] = self.trainer.datamodule.scaler_y.inverse_transform(
                        self.epoch_outputs[typ][task_i]["true"]
                    )
                    self.epoch_outputs[typ][task_i]["predictions"] = self.trainer.datamodule.scaler_y.inverse_transform(
                        self.epoch_outputs[typ][task_i]["predictions"]
                    )

        curr_epoch_metrics = get_metrics_and_figures(
            all_preds_and_true={typ: self.epoch_outputs[typ]},
            thresholds=0.5,
            pos_classes=self.pos_classes,
            figure_consumer=None,
            metrics_prefix=self.log_prefix,
        )

        # Reset the predictions and true values dict for the next epoch
        for task_i in self.task:
            self.epoch_outputs[typ][task_i]["predictions"] = []
            self.epoch_outputs[typ][task_i]["true"] = []

        # Log the metrics of the current epoch
        self.log_dict(curr_epoch_metrics, logger=True)

        # Log the average loss of the current epoch
        for k, v in self.curr_epoch_losses[typ].items():
            self.log(f"{self.log_prefix}{typ}/{k}", sum(v) / len(v))  # , step=self.current_epoch)
            v.clear()

        # Log the learning rate
        if typ == "train":
            if self.trainer.lr_scheduler_configs:
                lr = self.trainer.lr_scheduler_configs[0].scheduler.get_last_lr()[0]
            else:
                lr = self.optimizer.param_groups[0]['lr']
            self.log(f"{self.log_prefix}{typ}/lr", lr)

    def on_train_epoch_end(self):
        self._on_epoch_end("train")

    def on_validation_epoch_end(self):
        self._on_epoch_end("val")

    def on_test_epoch_end(self):
        self._on_epoch_end("test")

    def configure_optimizers(self):
        return self.optimizer


def get_model(model_config):
    """
    Convenience function initializing and returning the model based on the given model config.
    :param model_config: Dictionary containing "type": <model_type> and all required model parameters. Optionally contains "checkpoint_path": <ckpt_path_string> if a checkpoint should be loaded.
    :return: The model corresponding to the model_config
    """
    model_name = model_config["type"]
    ckpt_path = model_config.get("checkpoint_path", None)
    if model_name == PytorchModels.MLP.value:
        if ckpt_path is not None:
            print(f"Loading model from {model_config['checkpoint_path']}")
        model = MLP(input_size=model_config["input_size"],
                    hidden_sizes=model_config["hidden_sizes"],
                    output_size=model_config["output_size"],
                    batch_norm=model_config["batch_norm"],
                    activation_function=model_config["activation"],
                    dropout_input=model_config.get("dropout_input", model_config.get("dropout", .0)),
                    dropout_hidden=model_config.get("dropout_hidden", model_config.get("dropout", .0)))
    elif model_name == PytorchModels.LSTM.value:
        model = LSTM(
            input_size=model_config["input_size"],
            hidden_size=model_config["hidden_size"],
            num_layers=model_config.get("num_layers", 1),
            bias=model_config.get("bias", True),
            dropout=model_config.get("dropout", 0.0),
            bidirectional=model_config.get("bidirectional", False),
            proj_size=model_config["output_size"],
        )
    elif model_name == SKLearnModels.RF.value:
        model_config.pop("type", None)
        model_config.pop("input_size", None)
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(**model_config)
    elif model_name == SKLearnModels.HGBC.value:
        model_config.pop("type", None)
        model_config.pop("input_size", None)
        from sklearn.ensemble import HistGradientBoostingClassifier
        model = HistGradientBoostingClassifier(**model_config)
    else:
        raise NotImplementedError(f"The given model type is not implemented: {model_config['type']}")

    if ckpt_path is not None:
        state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
        print("Present keys in the checkpoint:", state_dict.keys())
        if model_config.get("checkpoint_layers", None) is not None:
            checkpoint_layers = model_config[
                "checkpoint_layers"]  # a list of expressions with wildcard, e.g., "model.encoder.*", to match the keys
            if isinstance(checkpoint_layers, str):
                checkpoint_layers = [checkpoint_layers]
            filtered_state_dict = {}
            for k, v in state_dict.items():
                for pattern in checkpoint_layers:
                    if fnmatch.fnmatch(k, pattern):
                        # Remove the prefix (e.g., "model.encoder.") to match the layer names in the current model
                        new_key = k.replace(pattern.split("*")[0], "")
                        filtered_state_dict[new_key] = v
            state_dict = filtered_state_dict
        assert state_dict, f"State dict is empty. Check the checkpoint_layers in the model config."
        model.load_state_dict(state_dict)

    return model


def get_LitModel(config: dict,
                 base_model: nn.Module,
                 criterion: nn.Module,
                 optimizer: torch.optim.Optimizer,
                 ckpt_path: Optional[str | Path] = None) -> LitModel:
    """
    Convenience function initializing and returning the LitModel based on the given config.
    :param config: The configuration dictionary of the experiment.
    :param base_model: The PyTorch model to train.
    :param criterion: The loss function to use.
    :param optimizer: The optimizer to use.
    :param ckpt_path: The path to a checkpoint to load the LitModel from.
    """
    litmodel_params = {"model": base_model,
                       "criterion": criterion,
                       "optimizer": optimizer,
                       "task": config["task"],
                       "pos_classes": config["training"].get("pos_classes", None)}
    if ckpt_path is not None:
        return LitModel.load_from_checkpoint(ckpt_path,
                                             **litmodel_params)
    else:
        return LitModel(**litmodel_params)
