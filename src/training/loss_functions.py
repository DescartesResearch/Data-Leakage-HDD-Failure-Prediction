import pandas as pd
import torch
import torch.nn as nn
import numpy as np


def apply_reduction(tensor: torch.Tensor, reduction: str) -> torch.Tensor:
    if reduction == "mean":
        return tensor.mean()
    elif reduction == "sum":
        return tensor.sum()
    elif reduction == "none":
        return tensor
    else:
        raise ValueError(f"Invalid reduction method: {reduction}")


class MSLELoss(nn.Module):
    """
    Implementation of the Mean Squared Logarithmic Error loss function.
    """

    def __init__(self, epsilon, reduction: str = "mean"):
        """
        Initialize the MSLELoss object.
        :param epsilon: A small offset value to prevent taking the log of zero.
        :param reduction: The reduction method to use. Can be "mean", "sum" or "none".
        """
        super(MSLELoss, self).__init__()
        self.epsilon = epsilon
        self.reduction = reduction

    def forward(self, y_pred, y_true):
        # Ensure that the tensors are of the same shape
        assert y_pred.shape == y_true.shape, f"Shapes of y_pred and y_true do not match: {y_pred.shape} vs {y_true.shape}"
        y_pred = torch.clamp(y_pred, min=0)

        # Calculate the log of the predictions and the true values
        log_pred = torch.log(y_pred + self.epsilon)
        log_true = torch.log(y_true + self.epsilon)

        # Calculate the mean squared error
        if self.reduction == "mean":
            return torch.mean((log_pred - log_true) ** 2)
        elif self.reduction == "sum":
            return torch.mean((log_pred - log_true) ** 2)
        elif self.reduction == "none":
            return (log_pred - log_true) ** 2
        else:
            raise ValueError(f"Invalid reduction method: {self.reduction}")


class WeightedSumLoss(nn.Module):
    """
    Custom loss function that calculates the weighted sum of multiple loss functions.
    """

    def __init__(self, loss_functions: list[nn.Module], alphas: list[float], *args, **kwargs):
        """
        Initialize the WeightedSumLoss object.
        :param loss_functions: A list of loss functions to use.
        :param alphas: A list of weights to apply to each loss function.
        """
        super().__init__(*args, **kwargs)
        self.loss_functions = loss_functions
        self.alphas = alphas

    def forward(self, y_pred, y_true):
        return [alpha * loss_fn(y_pred, y_true) for loss_fn, alpha, y_pred, y_true in
                zip(self.loss_functions, self.alphas, y_pred, y_true)]


def get_loss_function_with_params(loss_config: dict) -> nn.Module:
    """
    Get a loss function with the specified parameters.
    :param loss_config: A dictionary containing the name of the loss function (key: "name") and optionally its parameters (key: "params").
    :return: The loss function with the specified parameters.
    """
    name = loss_config["name"]
    params = loss_config.get("params", {})
    params["reduction"] = "none"
    if name in ["BCE", "BCELoss", "BinaryCrossEntropy", "BinaryCrossEntropyLoss"]:
        return torch.nn.BCELoss(**params)
    if name in ["BCEWithLogits", "BCEWithLogitsLoss", "BinaryCrossEntropyWithLogits",
                "BinaryCrossEntropyWithLogitsLoss"]:
        return torch.nn.BCEWithLogitsLoss(**params)
    elif name in ["CE", "CrossEntropy", "CrossEntropyLoss"]:
        return torch.nn.CrossEntropyLoss(**params)
    elif name in ["MSE", "MSELoss", "MeanSquaredError", "MeanSquaredErrorLoss"]:
        return torch.nn.MSELoss(**params)
    elif name in ["L1", "L1Loss", "MeanAbsoluteError", "MeanAbsoluteErrorLoss"]:
        return torch.nn.L1Loss(**params)
    elif name in ["MSLE", "MSLELoss", "MeanSquaredLogError", "MeanSquaredLogErrorLoss"]:
        return MSLELoss(**params)


def get_criterion(loss_config: dict | list[dict]):
    """
    Get the loss function with the specified parameters. If multiple loss functions are provided, return a WeightedSumLoss object.
    :param loss_config: A dictionary containing the name of the loss function (key: "name") and optionally its parameters (key: "params"). If multiple loss functions are provided, this should be a list of dictionaries, optionally with additional "alpha" keys for the weights.
    :return: The loss function with the specified parameters.
    """
    if isinstance(loss_config, list):
        loss_functions = [get_loss_function_with_params(loss_config_i) for loss_config_i in
                          loss_config]
        alphas = [loss_config_i.get("alpha", 1.0) for loss_config_i in loss_config]
        return WeightedSumLoss(loss_functions=loss_functions, alphas=alphas)
    else:
        return get_loss_function_with_params(loss_config)


def get_weights(strategy: str, class_counts: list[int] | pd.Series,
                device: torch._C.device = torch.device("cpu")) -> torch.Tensor:
    """
    Get the class weights for a given weighting strategy and the class counts.
    :param strategy: The weighting strategy to use, for example "balanced" or "equal".
    :param class_counts: The number of samples for each class in a list or pandas Series.
    :param device: The device to use for the tensor.
    :return: The class weights as a tensor, initialized on the given device.
    """
    num_classes = len(class_counts)
    num_samples = sum(class_counts)
    tensor_kwargs = {"dtype": torch.float32, "device": device}
    if strategy in ["inverse_class_frequency", "inverse_class_freq", "inverse_class_frequencies"]:
        weights = 1 / torch.tensor(class_counts, **tensor_kwargs)
    elif strategy in ["inverse_class_frequency_norm", "inverse_class_freq_norm", "inverse_class_frequencies_norm"]:
        weights = 1 / torch.tensor(class_counts, **tensor_kwargs)
        weights /= weights.sum()
    elif strategy == "balanced":
        weights = 1 / torch.tensor(class_counts, **tensor_kwargs)
        weights = weights * num_samples / num_classes
    elif strategy.startswith("effective_num_samples"):
        assert "beta=" in strategy, "Beta value must be provided for effective_num_samples strategy."
        beta = float(strategy.split("=")[1])
        weights = (1 - beta) / (1 - beta ** torch.tensor(class_counts, **tensor_kwargs))
    elif strategy == "equal":
        weights = torch.ones(num_classes, **tensor_kwargs)
    else:
        raise ValueError(f"Invalid weighting strategy: {strategy}")

    return weights
