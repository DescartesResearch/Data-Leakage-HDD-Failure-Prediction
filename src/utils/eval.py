import warnings
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
import scipy
from imblearn.metrics import geometric_mean_score
from scipy.stats import hmean
from sklearn.metrics import f1_score, roc_curve, precision_recall_curve, confusion_matrix, mean_squared_error, \
    mean_absolute_percentage_error, mean_absolute_error, r2_score, roc_auc_score, matthews_corrcoef, cohen_kappa_score, \
    log_loss, auc, brier_score_loss
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics._ranking import _binary_clf_curve, average_precision_score

from config.constants import Constants
from utils.misc import rec_defaultdict, flatten_dict, nested_key_exists, remove_len_zero_values, unflatten_dict


def data_leakage(test_data: pd.DataFrame, train_data: pd.DataFrame, group_col: str, datetime_col: str,
                 alpha: float = .0) -> float:
    """
    Implementation of the data leakage measure as defined in "Quantifying Data Leakage in Failure Prediction Tasks".
    As explained in the paper, the measure is solely based on the group and time information. Further features will
    not be considered in the calculation.
    :test_data: Pandas Dataframe with the test_data. Contains a group-identifying column <group_col> and a datetime column <datetime_col>
    :train_data: Pandas Dataframe with the train_data. Contains a group-identifying column <group_col> and a datetime column <datetime_col>
    :group_col: The column name of the group-identifying column.
    :datetime_col: The column name of the datetime column.
    :alpha: The weight of the leakage contribution for t_2 < t_1. Default is 0.
    :return: The normalized leakage value (float).
    """

    # Get the distances t_2 - t_1 for all pairs of observations (t_1, t_2) in the test and train data within the same group.
    distances: np.ndarray = get_distances_to_train_observations(test_data, train_data, group_col, datetime_col)

    # Get sum of all leakage contributions for case 1: t_2 >= t_1
    leakage = np.exp(-distances[distances >= 0]).sum()
    # Add sum of all leakage contributions for case 2: t_2 < t_1
    leakage += alpha * np.exp(distances[distances < 0]).sum()  # => unnormalized data leakage value L

    # Get normalization factor, that is, the maximum leakage value for the extreme case train_data = test_data
    distances: np.ndarray = get_distances_to_train_observations(test_data, test_data, group_col, datetime_col)
    leakage_worst_case = np.exp(-distances[distances >= 0]).sum()
    leakage_worst_case += alpha * np.exp(distances[distances < 0]).sum()

    return leakage / leakage_worst_case  # => normalized data leakage value


def get_distances_to_train_observations(test_data: pd.DataFrame, train_data: pd.DataFrame, group_col: str,
                                        datetime_col: str) -> np.ndarray:
    """
    For each observation in the test_data, finds the distance (in days) to the closest observation with the same group value in the train_data.
    If there is no such observation, the distance is np.inf.
    :param test_data:  The test data with a date column, group column and optionally further feature columns.
    :param train_data: The training data with a date column, group column and optionally further feature columns.
    :param group_col: A column defining different groups of observations in the dataset.
    :param datetime_col: The datetime column.
    :return: test_data with a new column day_diff.
    """
    test_data = test_data.reset_index(drop=False if test_data.index.name == datetime_col else True)
    train_data = train_data.reset_index(drop=False if train_data.index.name == datetime_col else True)
    test_data[datetime_col] = pd.to_datetime(test_data[datetime_col])
    train_data[datetime_col] = pd.to_datetime(train_data[datetime_col])

    merge_suffixes = ("_test", "_train")
    data_merged = pd.merge(left=test_data, right=train_data, on=group_col, how="inner", suffixes=merge_suffixes)
    day_diffs = (data_merged[datetime_col + "_train"] - data_merged[datetime_col + "_test"]).dt.days
    distances = day_diffs.values

    return distances


def aggregate_iterations(
        all_iterations_preds_and_true: list[dict[str, dict]],
        all_iterations_optimal_thresholds: Optional[list[dict[str, dict]]] = None,
) -> tuple[dict, Optional[dict]]:
    """
    Aggregates predictions/true values and found optimal thresholds of multiple iterations (e.g., with different seeds)
    to allow for an overall evaluation across all iterations. For predictions/true values, the values for each task
    (e.g., "binary" or "multiclass") are concatenated along the iterations. For the thresholds, the threshold values
    are augmented with the "begin_idx" and "end_idx" of the specific iteration.
    :param all_iterations_preds_and_true: List of dicts with structure {typ_i: {task_i: {"predictions": ..., "true": ...}}}
    :param all_iterations_optimal_thresholds: List of dicts with structure {typ_i: {task_i: {"binary_thresholds": {thresh_name_i: thresh_value_i, ...}}}}
    :return: Aggregated preds_and_true and thresholds. Same structure as the input dictionaries.
    """
    if all_iterations_optimal_thresholds is not None and len(all_iterations_preds_and_true) != len(
            all_iterations_optimal_thresholds):
        raise ValueError("The number of iterations in the two input lists must be equal!")
    if all_iterations_optimal_thresholds is None:
        all_iterations_optimal_thresholds = [{} for _ in range(len(all_iterations_preds_and_true))]
    n_iter = len(all_iterations_preds_and_true)
    # Check that all iterations have the same ds_types (train, val, test) and tasks
    ds_types = None
    tasks = None
    for i in range(0, len(all_iterations_preds_and_true)):
        if ds_types is None:
            ds_types = list(all_iterations_preds_and_true[i].keys())
        elif ds_types != list(all_iterations_preds_and_true[i].keys()):
            raise ValueError("All iterations must have the same ds_types!")
        for typ in ds_types:
            if tasks is None:
                tasks = list(all_iterations_preds_and_true[i][typ].keys())
            elif tasks != list(all_iterations_preds_and_true[i][typ].keys()):
                raise ValueError("All iterations must have the same tasks!")
    agg_preds_and_true = rec_defaultdict()
    agg_thresholds = rec_defaultdict()
    for task in tasks:
        for typ in ds_types:  # typ in [train, test, val]
            agg_preds_and_true[typ][task]["predictions"] = list()
            agg_preds_and_true[typ][task]["true"] = list()
            agg_thresholds[typ][task]["binary_thresholds"] = defaultdict(list)
            for i in range(n_iter):
                curr_iter_preds = all_iterations_preds_and_true[i][typ][task]["predictions"]
                curr_iter_true = all_iterations_preds_and_true[i][typ][task]["true"]
                agg_preds_and_true[typ][task]["predictions"].append(curr_iter_preds)
                agg_preds_and_true[typ][task]["true"].append(curr_iter_true)
                if nested_key_exists(all_iterations_optimal_thresholds[i],
                                     ["val", task, "binary_thresholds"]):
                    curr_binary_thresholds = all_iterations_optimal_thresholds[i]["val"][task][
                        "binary_thresholds"]
                    for thresh_name, thresh_value in curr_binary_thresholds.items():
                        begin_idx = 0 if len(agg_thresholds[typ][task]["binary_thresholds"][thresh_name]) == 0 else \
                            agg_thresholds[typ][task]["binary_thresholds"][thresh_name][-1][-1]
                        end_idx = begin_idx + curr_iter_preds.shape[0]
                        agg_thresholds[typ][task]["binary_thresholds"][thresh_name].append(
                            (thresh_value, begin_idx, end_idx))
            agg_preds_and_true[typ][task]["predictions"] = np.concatenate(
                agg_preds_and_true[typ][task]["predictions"], axis=0)
            agg_preds_and_true[typ][task]["true"] = np.concatenate(
                agg_preds_and_true[typ][task]["true"], axis=0)

    if all_iterations_optimal_thresholds is not None:
        agg_thresholds = remove_len_zero_values(agg_thresholds)
        return agg_preds_and_true, agg_thresholds
    else:
        return agg_preds_and_true, None


def find_thresholds(
        all_preds_and_true: dict,
        threshold_metrics: tuple[str, float | str] | list[tuple[str, float | str]],
        pos_classes: Optional[list] = None) -> dict:
    """
    Finds binary threshold w.r.t. the given threshold_metrics, for example ("MCC", "max"), which specifies that the
    binary threshold that maximizes the MCC metric should be found. For the multiclass-task, the classes specified by
    pos_classes are summarized to the positive class 1.
    The returned result always includes the FAR={0.01, 0.05, 0.1} and the standard 0.5 threshold.
    Only the validation predictions are used for determining the thresholds.
    :param all_preds_and_true: {typ_i: {task_i: {"predictions": ..., "true": ...}}}
    :param pos_classes: The classes in the multi-classification task that are to be summarized to the positive class 1
    :param threshold_metrics: A tuple or list of tuples with the format (metric_name, target_value), the target_value may be "max"/"min" or a float
    :return: {"val": {task_i: {"binary_thresholds": {thresh_name_i: thresh_value_i, ...}, ...}} Example: {"val": {"multiclass": {"binary_thresholds": {"MCC_max": 0.6}, {"0.5": 0.5}, {"FAR_0.01": 0.45}, ...}}, {"binary": {"binary_thresholds": {"MCC_max": 0.55}, {"0.5": 0.5}, {"FAR_0.01": 0.47}, ...}}}
    """
    if not isinstance(threshold_metrics, list):
        threshold_metrics = [threshold_metrics]
    if "multiclass" in all_preds_and_true["val"].keys() and pos_classes is None:
        raise ValueError("pos_classes must be specified for multi-classification tasks!")
    threshold_configs = [(get_metric_func(threshold_metric[0], labels=[0, 1], metric_kwargs={}),
                          threshold_metric[1],
                          "_".join(threshold_metric)
                          ) for threshold_metric in threshold_metrics]

    # add standard 0.5 threshold and FAR={0.01, 0.05, 0.1} as threshold configs
    threshold_configs.append((None, 0.5, "0.5"))
    threshold_configs.extend(
        [(get_metric_func("FAR", labels=[0, 1]), FAR_tgt_value, f"FAR_{FAR_tgt_value}") for FAR_tgt_value in
         [0.01, 0.05, 0.1]])

    optimal_thresholds = rec_defaultdict()
    for task_i in all_preds_and_true["val"].keys():
        if task_i == "binary":
            y_true_binary = all_preds_and_true["val"][task_i]["true"]
            y_pred_binary_logits = all_preds_and_true["val"][task_i]["predictions"]
        elif task_i == "multiclass":
            y_true_binary = multiclass_labels_to_binary_labels(all_preds_and_true["val"][task_i]["true"],
                                                               pos_classes=pos_classes)
            y_pred_binary_logits = multiclass_logits_to_binary_logits(all_preds_and_true["val"][task_i]["predictions"],
                                                                      pos_classes=pos_classes)
        else:
            continue
        for metric_func, target, thresh_name in threshold_configs:
            if metric_func is not None:
                curr_thresh = get_best_binary_threshold(y_true=y_true_binary,
                                                        y_score=y_pred_binary_logits,
                                                        metric_func=metric_func,
                                                        target=target,
                                                        )
            else:
                curr_thresh = target
            optimal_thresholds["val"][task_i]["binary_thresholds"][thresh_name] = curr_thresh

    return optimal_thresholds


def get_metrics_and_figures(all_preds_and_true: dict,
                            thresholds: dict | float = None,
                            pos_classes: list = None,
                            figure_consumer: Optional[callable] = None,
                            metrics_prefix: str = Constants.LOGGING_METRICS_PREFIX,
                            figures_prefix: str = Constants.LOGGING_FIGURES_PREFIX) -> dict:
    """
    Calculates and returns all relevant metrics for the given predictions and true values.
    For the tasks "binary" and "multiclass", thresholds must be specified.
    For the "multiclass" task, the positive classes must be specified additionally.
    If a figure_consumer callable is provided, figures are created and passed to the consumer.
    :param all_preds_and_true: {typ_i: {task_i: {"predictions": ..., "true": ...}}}
    :param thresholds: The thresholds for the binary classification tasks. If a float is given, the threshold is used for all metrics. Otherwise, thresholds must have this structure: {typ_i: {task_i: {"binary_thresholds": {thresh_name_i: thresh_value_i, ...}}}. If for train/test no thresholds are specified, the thresholds for the validation set are used.
    :param pos_classes: The classes in the multi-classification task that are to be summarized to the positive class 1.
    :param figure_consumer: A callable (figure: matplotlib.figure, figure_name: str) => None. If figure_consumer is None, no figures are created.
    :param metrics_prefix: The prefix for the metrics in the log.
    :param figures_prefix: The prefix for the figures in the log.
    :return: A dictionary with all metrics.
    """
    if not metrics_prefix.endswith("/"):
        metrics_prefix += "/"
    if not figures_prefix.endswith("/"):
        figures_prefix += "/"
    sns.set_style('white')
    sns.set_context("paper", font_scale=2)
    if isinstance(thresholds, float):
        thresholds = {"val":
            {
                "multiclass": {"binary_thresholds": {str(thresholds): thresholds}},
                "binary": {"binary_thresholds": {str(thresholds): thresholds}}}
        }
    all_metrics = {}
    for typ in all_preds_and_true.keys():
        for task_i in all_preds_and_true[typ].keys():
            y_true = all_preds_and_true[typ][task_i]["true"]
            y_hat = all_preds_and_true[typ][task_i]["predictions"]
            if task_i == "multiclass":
                if thresholds is None:
                    raise ValueError("thresholds must be specified for multi-classification tasks!")
                if pos_classes is None:
                    raise ValueError("pos_classes must be specified for multi-classification tasks!")
                metrics = get_multi_classification_metrics(y_true, y_hat)
                if figure_consumer is not None:
                    EvaluationPlots(
                        log_prefix=f"{figures_prefix}{typ}/{task_i}",
                        y_hat=y_hat,
                        y_true=y_true,
                        task=task_i,
                        figure_consumer=figure_consumer
                    ).plot_all()

                if typ in thresholds.keys():
                    binary_thresholds = thresholds[typ][task_i][
                        "binary_thresholds"]  # {thresh_name_i: thresh_value_i, ...}
                else:
                    binary_thresholds = thresholds["val"][task_i][
                        "binary_thresholds"]  # {thresh_name_i: thresh_value_i, ...}
                _thresh_val = list(binary_thresholds.values())[0]

                if isinstance(_thresh_val, list):
                    iter_intervals = [t[1:] for t in _thresh_val]
                else:
                    iter_intervals = None

                # print(f"typ={typ}, task={task_i}, iter_intervals={iter_intervals}")

                y_hat_binary = multiclass_logits_to_binary_logits(
                    y_score=y_hat,
                    pos_classes=pos_classes)
                y_true_binary = multiclass_labels_to_binary_labels(y_true, pos_classes=pos_classes)
                # get threshold-independent (-> threshold=None) metrics:
                metrics.update({f"cg/{k}": v for k, v in
                                get_binary_classification_metrics(
                                    y_true_binary,
                                    y_hat_binary,
                                    threshold=None).items()
                                })
                if figure_consumer is not None:
                    EvaluationPlots(
                        figure_consumer=figure_consumer,
                        log_prefix=f"{figures_prefix}{typ}/{task_i}/cg",
                        y_hat=y_hat_binary,
                        y_true=y_true_binary,
                        task="binary",
                        binary_threshold=None,
                        iter_intervals=iter_intervals,
                    ).plot_all()
                # get threshold-dependent metrics:
                for thresh_name, thresh_value in binary_thresholds.items():
                    metrics.update({f"cg/thresh={thresh_name}/{k}": v for k, v in
                                    get_binary_classification_metrics(
                                        y_true_binary,
                                        y_hat_binary,
                                        threshold=thresh_value).items()
                                    })
                    if figure_consumer is not None:
                        EvaluationPlots(
                            figure_consumer=figure_consumer,
                            log_prefix=f"{figures_prefix}{typ}/{task_i}/cg/thresh={thresh_name}",
                            y_hat=y_hat_binary,
                            y_true=y_true_binary,
                            task="binary",
                            binary_threshold=thresh_value,
                            iter_intervals=iter_intervals,
                        ).plot_all()
            elif task_i == "binary":
                assert thresholds is not None

                if typ in thresholds.keys():
                    binary_thresholds = thresholds[typ][task_i][
                        "binary_thresholds"]  # {thresh_name_i: thresh_value_i, ...}
                else:
                    binary_thresholds = thresholds["val"][task_i][
                        "binary_thresholds"]  # {thresh_name_i: thresh_value_i, ...}
                _thresh_val = list(binary_thresholds.values())[0]

                if isinstance(_thresh_val, list):
                    iter_intervals = [t[1:] for t in _thresh_val]
                else:
                    iter_intervals = None

                metrics = {}

                # get threshold-independent (-> threshold=None) metrics:
                metrics.update({f"{k}": v for k, v in
                                get_binary_classification_metrics(
                                    y_true,
                                    y_hat,
                                    threshold=None).items()
                                })
                if figure_consumer is not None:
                    EvaluationPlots(
                        figure_consumer=figure_consumer,
                        log_prefix=f"{figures_prefix}{typ}/{task_i}",
                        y_hat=y_hat,
                        y_true=y_true,
                        task=task_i,
                        binary_threshold=None,
                        iter_intervals=iter_intervals,
                    ).plot_all()

                # get threshold-dependent metrics:
                for thresh_name, thresh_value in binary_thresholds.items():
                    metrics.update({f"thresh={thresh_name}/{k}": v for k, v in
                                    get_binary_classification_metrics(
                                        y_true,
                                        y_hat,
                                        threshold=thresh_value).items()
                                    })
                    if figure_consumer is not None:
                        EvaluationPlots(
                            figure_consumer=figure_consumer,
                            log_prefix=f"{figures_prefix}{typ}/{task_i}/thresh={thresh_name}",
                            y_hat=y_hat,
                            y_true=y_true,
                            task=task_i,
                            binary_threshold=thresh_value,
                            iter_intervals=iter_intervals,
                        ).plot_all()
            elif task_i == "regression":
                metrics = get_regression_metrics(y_true, y_hat)
                if figure_consumer is not None:
                    EvaluationPlots(
                        figure_consumer=figure_consumer,
                        log_prefix=f"{figures_prefix}{typ}/{task_i}",
                        y_hat=y_hat,
                        y_true=y_true,
                        task=task_i,
                    ).plot_all()
            elif task_i == "reconstruction":
                if len(y_true.shape) == 3:
                    _y_true = y_true.reshape(-1, y_true.shape[1] * y_true.shape[2])
                    _y_hat = y_hat.reshape(-1, y_hat.shape[1] * y_hat.shape[2])
                else:
                    _y_true = y_true
                    _y_hat = y_hat
                metrics = get_regression_metrics(_y_true, _y_hat)
                if figure_consumer is not None:
                    EvaluationPlots(
                        figure_consumer=figure_consumer,
                        log_prefix=f"{figures_prefix}{typ}/{task_i}",
                        y_hat=y_hat,
                        y_true=y_true,
                        task=task_i,
                    ).plot_all()
            else:
                raise NotImplementedError()

            for k, v in metrics.items():
                all_metrics[f"{metrics_prefix}{typ}/{task_i}/{k}"] = v

    return all_metrics


def get_metric_func(metric_name: str, labels: list, metric_kwargs: dict = None) -> callable:
    """
    Returns the metric function for the given metric_name.
    :param metric_name: The name of the metric.
    :param labels: The labels for the classification task.
    :param metric_kwargs: Additional keyword arguments for the metric function.
    :return: The metric function (y_true, y_pred) -> float.
    """
    if metric_kwargs is None:
        metric_kwargs = {}
    if metric_name == "MCC":
        return lambda y_true, y_pred: matthews_corrcoef(y_true, y_pred, **metric_kwargs)
    elif metric_name == "cohen_kappa":
        return lambda y_true, y_pred: cohen_kappa_score(y_true, y_pred, labels=labels, **metric_kwargs)
    elif metric_name == "macro_f1_score":
        return lambda y_true, y_pred: f1_score(y_true, y_pred, labels=labels, average="macro", **metric_kwargs)
    elif metric_name == "FAR":
        def FAR(y_true, y_pred):
            conf_matrix = confusion_matrix(y_true=y_true, y_pred=y_pred, labels=labels, **metric_kwargs)
            tn, fp, fn, tp = conf_matrix.ravel()
            far = fp / (fp + tn) if fp + tn > 0 else 0
            return far

        return FAR
    elif metric_name == "G_mean":
        return lambda y_true, y_pred: geometric_mean_score(y_true=y_true, y_pred=y_pred, labels=labels, pos_label=1,
                                                           **metric_kwargs)
    else:
        raise ValueError(f"Unknown metric name: {metric_name}")


def get_classification_metrics(tp: int, tn: int, fp: int, fn: int) -> dict:
    """
    Calculate classification metrics based on the confusion matrix components.
    :param tp: True positives
    :param tn: True negatives
    :param fp: False positives
    :param fn: False negatives
    :return: Dictionary with classification metrics.
    """
    # to avoid numerical overflows - doesn't affect the results below as max_val factors out
    if (tp + fn) == 0:
        warnings.warn(f"No positive samples! Metrics ill-defined. fp={fp}, tn={tn}")
    if (tn + fp) == 0:
        warnings.warn(f"No negative samples! Metrics ill-defined. tp={tp}, fn={fn}")
    max_val = max(tp, tn, fp, fn)
    tp = tp / max_val
    tn = tn / max_val
    fp = fp / max_val
    fn = fn / max_val
    precision = tp / (tp + fp) if tp + fp > 0 else 0
    recall = tp / (tp + fn) if tp + fn > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0
    specificity = tn / (tn + fp) if tn + fp > 0 else 0
    sensitivity = fdr = recall
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    g_mean = np.sqrt(sensitivity * specificity)
    far = fp / (fp + tn) if fp + tn > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    hmcp = hmean([precision, recall, npv, specificity])
    mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) if (tp + fp) * (tp + fn) * (
            tn + fp) * (tn + fn) > 0 else 0
    return {
        "precision": precision,
        "recall": recall,
        "F1": f1,
        "specificity": specificity,
        "sensitivity": sensitivity,
        "G_mean": g_mean,
        "FDR": fdr,
        "FAR": far,
        "accuracy": accuracy,
        "NPV": npv,
        "HMCP": hmcp,
        "MCC": mcc,
    }


def get_binary_classification_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray,
                                      threshold: float | list[tuple[float, int, int]] | None) -> dict:
    """
    Calculate classification metrics for binary classification tasks. If a threshold is given, threshold-dependent
    metrics like precision, recall, F1, G_mean, accuracy are calculated. Otherwise
    """
    if threshold is not None:
        y_pred = binary_logits_to_labels(y_pred_logits, threshold=threshold)
        conf_matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tp = conf_matrix[1, 1]
        tn = conf_matrix[0, 0]
        fp = conf_matrix[0, 1]
        fn = conf_matrix[1, 0]
        metrics = get_classification_metrics(tp, tn, fp, fn)
        metrics["macro_avg_F1"] = f1_score(y_true, y_pred, average="macro")
        return metrics
    else:
        metrics = dict()
        metrics["AUROC"] = roc_auc_score(y_true, y_pred_logits, labels=[0, 1])
        metrics["AUPRC"] = average_precision_score(y_true, y_pred_logits)
        metrics["log_loss"] = log_loss(y_true, y_pred_logits, labels=[0, 1])
        metrics["brier_score_loss"] = brier_score_loss(y_true, y_pred_logits, pos_label=1)
        return metrics


def get_multi_classification_metrics(y_true: np.ndarray, y_pred_logits: np.ndarray) -> dict:
    # convert logits to predictions
    y_pred = np.argmax(y_pred_logits, axis=1)
    num_classes = y_pred_logits.shape[1]
    conf_matrix = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    metrics = {}
    macro_averages = defaultdict(list)
    tps, fps, fns, tns = [], [], [], []
    for i in range(num_classes):
        tp = conf_matrix[i, i]
        fp = np.sum(conf_matrix[:, i]) - tp
        fn = np.sum(conf_matrix[i, :]) - tp
        tn = np.sum(conf_matrix) - tp - fp - fn
        tps.append(tp)
        fps.append(fp)
        fns.append(fn)
        tns.append(tn)
        classification_metrics = get_classification_metrics(tp, tn, fp, fn)
        metrics.update({f"class_{i}/" + k: v for k, v in classification_metrics.items() if
                        k not in ["FDR", "FAR"]})  # we do not need FDR and FAR for each class
        for k, v in classification_metrics.items():
            macro_averages[k].append(v) if k not in ["FDR", "FAR"] else None
    macro_averages = {k: np.mean(v) for k, v in macro_averages.items()}
    metrics.update({"macro_avg/" + k: v for k, v in macro_averages.items()})
    metrics.update(
        {"micro_avg/" + k: v for k, v in get_classification_metrics(sum(tps), sum(tns), sum(fps), sum(fns)).items() if
         k not in ["FDR", "FAR"]})
    metrics["macro_avg/F1_sanity_check"] = f1_score(y_true, y_pred, average="macro", labels=list(range(num_classes)))
    metrics["micro_avg/F1_sanity_check"] = f1_score(y_true, y_pred, average="micro", labels=list(range(num_classes)))
    metrics["multi/cohen_kappa"] = cohen_kappa_score(y_true, y_pred, labels=list(range(num_classes)))
    metrics["multi/MCC"] = matthews_corrcoef(y_true, y_pred)
    metrics["multi/log_loss"] = log_loss(y_true, scipy.special.softmax(y_pred_logits, axis=1),
                                         labels=list(range(num_classes)),
                                         normalize=False)
    return metrics


def get_regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    return {
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "MAPE": mape,
        "R2": r2
    }


def multiclass_logits_to_binary_logits(y_score: np.ndarray, pos_classes: list[int]):
    return np.clip(
        scipy.special.softmax(y_score, axis=1)[:, pos_classes].sum(axis=1),
        0.0, 1.0)


def multiclass_logits_to_binary_labels(y_score: np.ndarray, pos_classes: list[int],
                                       threshold: float | list[tuple[float, int, int]]):
    y_score_binary = multiclass_logits_to_binary_logits(y_score, pos_classes)
    return binary_logits_to_labels(y_score_binary, threshold)


def binary_logits_to_labels(y_score: np.ndarray, threshold: float | list[tuple[float, int, int]]):
    if isinstance(threshold, list):
        coverage: np.ndarray = np.zeros_like(y_score)
        for t in threshold:
            coverage[t[1]:t[2]] += 1
        assert (coverage == 1).all()
    y_pred = (y_score >= threshold).astype(int) if not isinstance(threshold, list) else np.concatenate(
        [(y_score[t[1]:t[2]] >= t[0]) for t in threshold], axis=0).astype(int)
    return y_pred


def multiclass_labels_to_binary_labels(y_multiclass: np.ndarray, pos_classes: list[int]) -> np.ndarray:
    return np.isin(y_multiclass, pos_classes).astype(int)


def is_better_than(x: Optional[float], y: Optional[float], target: str | float | int) -> bool:
    if x is None:
        return False
    elif y is None:
        return True
    elif isinstance(target, str):
        return x > y if target == "max" else x < y
    else:
        return True if np.abs(x - target) < np.abs(y - target) else False


def get_best_binary_threshold(y_true: np.ndarray,
                              y_score: np.ndarray,
                              target: str | float,
                              metric_func: callable,
                              metric_func_kwargs: dict = None) -> float:
    """
    Find the optimal binary threshold for the binary classification of the labels.
    :param y_true: The true labels.
    :param y_score: The predicted scores.
    :param target: The target value for the metric function. Can be "min", "max" or a specific target value.
    :param metric_func: The metric function to optimize.
    :param metric_func_kwargs: Additional keyword arguments for the metric function.
    :return: Optimal threshold for the given metric_func and target ("min", "max" or specific target value (float))
    """
    if isinstance(target, str):
        target = target.lower()
        assert target in ["min", "max"]

    if metric_func_kwargs is None:
        metric_func_kwargs = {}

    fps, tps, thresholds = _binary_clf_curve(
        y_true, y_score, pos_label=1, sample_weight=None
    )

    if len(fps) > 2:
        # Drop thresholds corresponding to points where true positives (tps)
        # do not change from the previous or subsequent point. This will keep
        # only the first and last point for each tps value. All points
        # with the same tps value have the same recall and thus x coordinate.
        # They appear as a vertical line on the plot.
        optimal_idxs = np.where(
            np.concatenate(
                [[True], np.logical_or(np.diff(tps[:-1]), np.diff(tps[1:])), [True]]
            )
        )[0]
        fps = fps[optimal_idxs]
        tps = tps[optimal_idxs]
        thresholds = thresholds[optimal_idxs]

    best_threshold = None
    best_metric_value = None
    for t in thresholds:
        y_pred = (y_score >= t).astype(int)
        metric_value = metric_func(y_true, y_pred, **metric_func_kwargs)
        if is_better_than(metric_value, best_metric_value, target):
            best_metric_value = metric_value
            best_threshold = t
    return best_threshold


def aggregate_figures_and_metrics(
        aggregate_metrics: bool = True,
        metric_dicts: Optional[list[dict]] = None,
        test_only: bool = False,
        all_preds_and_true: Optional[list[dict]] = None,
        all_thresholds: Optional[list[dict]] = None,
        figure_consumer: Optional[callable] = None,
        add_concat_metrics: Optional[bool] = None,
        pos_classes: Optional[list[int]] = None) -> dict:
    """
    Aggregate figures and metrics from multiple iterations for the given run.

    :param metric_dicts: List of dictionaries with the structure {typ_i: {task_i: <metric_dict>}}
    :param test_only: Iff true, only aggregate the test metrics and figures
    :param aggregate_figures: Iff True, aggregates figures
    :param aggregate_metrics: Iff True, aggregates metrics
    :param add_concat_metrics: Iff True, also calculates metrics based on the concatenated predictions.
    :param all_preds_and_true: List of dictionaries with the structure {typ_i: {task_i: {"predictions": ..., "true": ...}}}
    :return: A dictionary with the aggregated metrics
    """
    if figure_consumer is not None or add_concat_metrics:
        # Log figures and metrics based on the concatenated predictions
        if all_preds_and_true is None:
            raise ValueError("all_preds_and_true must be provided if aggregate_figures or add_concat_metrics is True.")
        if len(all_preds_and_true) != len(metric_dicts):
            raise ValueError(
                "The number of iterations in all_preds_and_true must be equal to the number of iterations in metric_dicts.")
        if add_concat_metrics is None:
            add_concat_metrics = True
        agg_preds_and_true, agg_thresholds = aggregate_iterations(all_preds_and_true,
                                                                  all_thresholds)
        concat_metrics = get_metrics_and_figures(agg_preds_and_true,
                                                 agg_thresholds,
                                                 pos_classes=pos_classes,
                                                 figure_consumer=figure_consumer,
                                                 figures_prefix=Constants.LOGGING_FIGURES_PREFIX,
                                                 metrics_prefix=f"{Constants.LOGGING_METRICS_PREFIX}/concat")
    else:
        concat_metrics = {}

    global_results = defaultdict(list)
    aggregated_metrics = dict()
    if aggregate_metrics:
        # write the metric values from the k iterations into lists of length k
        for curr_metrics in list(map(flatten_dict, metric_dicts)):
            for k, v in curr_metrics.items():
                global_results[k].append(v)

        # aggregate the values of the lists with different aggregation methods
        for k, value_list in global_results.items():
            if len(value_list) != len(metric_dicts):
                raise ValueError(f"Length of value list for key {k} is not equal to the number of iterations.")
            for agg_method, agg_method_name in [(np.mean, "mean"), (np.median, "median"),
                                                (lambda a: np.std(a, ddof=1), "std")]:
                aggregated_metrics[f"{Constants.LOGGING_METRICS_PREFIX}/{agg_method_name}/{k}"] = agg_method(value_list)

    return unflatten_dict({**aggregated_metrics, **concat_metrics})


class EvaluationPlots:
    """
    Class for creating evaluation plots for regression, multiclass and binary classification tasks.
    """

    def __init__(self,
                 log_prefix: str,
                 y_hat: np.ndarray,
                 y_true: np.ndarray,
                 task: str,
                 figure_consumer: Optional[callable] = None,
                 binary_threshold: float | list[tuple[float, int, int]] = None,
                 iter_intervals: list[tuple[int, int]] = None):
        """
        :param log_prefix: The log prefix prepended to the figure names when passed to the figure consumer.
        :param y_hat: The predicted values.
        :param y_true: The true values.
        :param task: The task of the model, for example "regression", "multiclass" or "binary".
        :param figure_consumer: A callable that takes a matplotlib figure and a name as arguments and returns None.
        :param binary_threshold: The binary threshold for the binary classification plots.
        :param iter_intervals: The intervals for plotting the mean ROC- and PR-curves for multiple iterations. The intervals indicate the start and end index of the predictions for each iteration.
        """
        self.figure_consumer = figure_consumer
        self.log_prefix = log_prefix if log_prefix.endswith("/") else log_prefix + "/"
        self.y_hat = y_hat
        self.y_true = y_true
        self.task = task
        self.binary_threshold = binary_threshold
        self.iter_intervals = iter_intervals
        self.eval_data = {}

    def _show_and_close(self, fig, name):
        # if a figure consumer is present, pass to it, otherwise simply show with name as title
        if self.figure_consumer is not None:
            self.figure_consumer(fig, self.log_prefix + name)
        else:
            fig.suptitle(name)
            plt.show()
        plt.close(fig)

    def get_residuals(self):
        if self.task not in ["regression", "reconstruction"]:
            return None
        if "residuals" in self.eval_data:
            return self.eval_data["residuals"]
        else:
            self.eval_data["residuals"] = self.y_true - self.y_hat
            return self.get_residuals()

    def residual_plot(self):
        if self.task not in ["regression"]:
            return
        fig, ax = plt.subplots()
        sns.histplot(self.get_residuals(), bins="auto", ax=ax)
        ax.set_xlabel("Residuals: True - Predicted")
        ax.set_ylabel("Frequency")
        ax.set_title("Residuals Histogram")
        self._show_and_close(fig, "residuals")

    def histogram_plots(self):
        if self.task not in ["regression"]:
            return
        fig, ax = plt.subplots()
        sns.histplot(self.y_true, ax=ax)
        ax.set_title("True values")
        self._show_and_close(fig, "true_values_histogram")

        fig, ax = plt.subplots()
        sns.histplot(self.y_hat, ax=ax)
        ax.set_title("Predicted values")
        self._show_and_close(fig, "pred_values_histogram")

    def pred_vs_true_scatter_plot(self):
        if self.task not in ["regression"]:
            return
        min_value = int(min(self.y_true.min(), self.y_hat.min()))
        max_value = int(max(self.y_true.max(), self.y_hat.max())) + 1
        fig, ax = plt.subplots()
        ax.scatter(self.y_true, self.y_hat)
        ax.plot([min_value, max_value], [min_value, max_value], color="red", linestyle="--")
        ax.set_xlabel("True")
        ax.set_ylabel("Predicted")
        ax.set_title("Pred vs. True")
        self._show_and_close(fig, "pred_vs_true_scatter_plot")

    def binwise_mae_mse_plots(self):
        if self.task not in ["regression"]:
            return
        bin_width = 7  # 1 week
        bins = range(0, self.y_true.max().astype(int) + bin_width, bin_width)
        residuals_binned = []
        bin_sizes = []
        for i in range(len(bins) - 1):
            mask = (self.y_true >= bins[i]) & (self.y_true < bins[i + 1])
            residuals_binned.append(self.get_residuals()[mask])
            bin_sizes.append(mask.sum().item())
        # Get the MAE and RMSE for each bin
        mae_binned = [np.mean(np.abs(res)).item() if len(res) > 0 else np.nan for res in residuals_binned]
        rmse_binned = [np.sqrt(np.mean(res ** 2)).item() if len(res) > 0 else np.nan for res in residuals_binned]
        # Plot a bar plot for the MAE for each bin and write the bin sizes above the bars

        fig, ax = plt.subplots()
        sns.barplot(x=range(1, len(mae_binned) + 1), y=mae_binned, ax=ax)
        for i, bin_size in enumerate(bin_sizes):
            if bin_size == 0:
                continue
            ax.text(i, mae_binned[i] + 0.1, str(bin_size), color='black', ha="center")
        ax.set_xlabel("Weeks to Failure")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("MAE vs Week")
        self._show_and_close(fig, "/mae_vs_week")

        fig, ax = plt.subplots()
        sns.barplot(x=range(1, len(rmse_binned) + 1), y=rmse_binned, ax=ax)
        for i, bin_size in enumerate(bin_sizes):
            if bin_size == 0:
                continue
            ax.text(i, rmse_binned[i] + 0.1, str(bin_size), color='black', ha="center")
        ax.set_xlabel("Weeks to Failure")
        ax.set_ylabel("Root Mean Squared Error")
        ax.set_title("RMSE vs Week")
        self._show_and_close(fig, "rmse_vs_week")

    def multi_classification_plots(self):
        if self.task not in ["multiclass"]:
            return
        y_pred = np.argmax(self.y_hat, axis=1)
        num_classes = self.y_hat.shape[1]

        # Plot multiclass confusion matrix
        fig, ax = plt.subplots()
        plot_confusion_matrix(self.y_true, y_pred, labels=list(range(num_classes)), normalize=False, ax=ax)
        self._show_and_close(fig, "confusion_matrix_abs")

        fig, ax = plt.subplots()
        plot_confusion_matrix(self.y_true, y_pred, labels=list(range(num_classes)), normalize=True, ax=ax)
        self._show_and_close(fig, "confusion_matrix_rel")

    def binary_classification_plots(self):
        if self.task not in ["binary"]:
            return
        # threshold-dependent plots, namely the confusion matrix
        if self.binary_threshold is not None:
            y_pred = binary_logits_to_labels(self.y_hat, threshold=self.binary_threshold)
            num_classes = 2

            fig, ax = plt.subplots()
            plot_confusion_matrix(self.y_true, y_pred, labels=list(range(num_classes)), normalize=False, ax=ax)
            self._show_and_close(fig, "confusion_matrix_abs")

            fig, ax = plt.subplots()
            plot_confusion_matrix(self.y_true, y_pred, labels=list(range(num_classes)), normalize=True, ax=ax)
            self._show_and_close(fig, "confusion_matrix_rel")

        # threshold-independent plots like ROC and PR curves
        else:
            fig, ax = plt.subplots()
            if self.iter_intervals is None:
                fpr, tpr, _ = roc_curve(self.y_true, self.y_hat)
                auc_score = auc(fpr, tpr)
                # Plot the ROC curve
                ax.plot(fpr, tpr)
                ax.plot([0, 1], [0, 1], linestyle="--")
                ax.set_xlabel("False Positive Rate (FAR)")
                ax.set_ylabel("True Positive Rate (FDR)")
                ax.set_title(f"ROC Curve (AUC={auc_score})")
            else:
                plot_mean_roc_curve(self.y_true, self.y_hat, intervals=self.iter_intervals, ax=ax)
            self._show_and_close(fig, "roc_curve")

            fig, ax = plt.subplots()
            if self.iter_intervals is None:
                prec, rec, _ = precision_recall_curve(self.y_true, self.y_hat)
                prec, rec = prec[::-1], rec[::-1]
                ap_score = average_precision_score(self.y_true, self.y_hat)
                # No skill line
                # precision_wo_skill, recall_wo_skill, _ = precision_recall_curve(self.y_true, np.ones_like(self.y_true))
                # precision_wo_skill, recall_wo_skill = precision_wo_skill[::-1], recall_wo_skill[::-1]
                pos_prevalence = self.y_true.sum() / len(self.y_true)
                ax.plot(rec, prec)
                ax.plot([0, 1], [pos_prevalence, pos_prevalence], linestyle="--")
                ax.set_xlabel("Recall")
                ax.set_ylabel("Precision")
                ax.set_title(f"Precision-Recall Curve (AP={ap_score})")
            else:
                plot_mean_pr_curve(self.y_true, self.y_hat, intervals=self.iter_intervals, ax=ax)
            self._show_and_close(fig, "pr_curve")

    def reconstruction_error_plot(self):
        if self.task not in ["reconstruction"] or len(self.y_true.shape) != 3:
            return
        residuals = self.get_residuals()
        # Shape is N x seq_len x C
        mae = np.mean(np.abs(residuals), axis=(0, 2))
        std = np.std(residuals, axis=(0, 2))
        fig, ax = plt.subplots()
        ax.plot(mae)
        ax.fill_between(range(len(mae)), mae - std, mae + std, alpha=0.2)
        ax.set_xlabel("Time")
        ax.set_ylabel("Mean Absolute Error")
        ax.set_title("Reconstruction Error")
        self._show_and_close(fig, "reconstruction_error")

    def residuals_scatter_plot(self):
        """
        Plots a line with slope 1 and intercept 0 and the residuals as a scatter plot around this line.
        In a perfect model, the residuals should lie on or close to the line.
        :return:
        """
        if self.task not in ["regression"]:
            return
        fig, ax = plt.subplots()
        residuals = self.get_residuals()
        ax.scatter(self.y_true, residuals)
        ax.plot([min(self.y_true), max(self.y_true)], [0, 0], color="red")
        ax.set_xlabel("True values")
        ax.set_ylabel("Residuals: True - Predicted")
        ax.set_title("Residuals Scatter Plot")
        self._show_and_close(fig, "residuals_scatter_plot")

    def plot_all(self):
        self.residual_plot()
        self.histogram_plots()
        self.binwise_mae_mse_plots()
        self.residuals_scatter_plot()
        self.pred_vs_true_scatter_plot()
        self.binary_classification_plots()
        self.multi_classification_plots()


def plot_mean_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, intervals: list[tuple[int, int]], ax):
    coverage = np.zeros_like(y_true)
    for begin, end in intervals:
        coverage[begin:end] += 1
    assert (coverage == 1).all()

    fpr_list = []
    tpr_list = []
    auc_list = []

    for begin, end in intervals:
        fpr, tpr, _ = roc_curve(y_true[begin:end], y_scores[begin:end])
        fpr_list.append(fpr)
        tpr_list.append(tpr)
        auc_list.append(auc(fpr, tpr))

    mean_auc = np.mean(auc_list)
    std_auc = np.std(auc_list, ddof=1)

    # Interpolation step
    mean_fpr = np.linspace(0, 1, 1000)
    tpr_list_interpolated = list()
    for fpr, tpr in zip(fpr_list, tpr_list):
        tpr_list_interpolated.append(np.interp(mean_fpr, fpr, tpr).reshape(1, -1))
    tpr_interpolated = np.concatenate(tpr_list_interpolated, axis=0)
    mean_tpr = np.mean(tpr_interpolated, axis=0)
    std_tpr = np.std(tpr_interpolated, axis=0, ddof=1)

    # Plot the mean ROC curve
    ax.plot(mean_fpr, mean_tpr)
    ax.fill_between(mean_fpr, mean_tpr - std_tpr, mean_tpr + std_tpr, color='blue', alpha=0.2)
    ax.plot([0, 1], [0, 1], linestyle="--")
    ax.set_xlabel("False Positive Rate (FAR)")
    ax.set_ylabel("True Positive Rate (FDR)")
    ax.set_title(f"Mean ROC Curve (AUC = {mean_auc:.2f} ± {std_auc:.2f})")


def plot_mean_pr_curve(y_true: np.ndarray, y_scores: np.ndarray, intervals: list[tuple[int, int]], ax):
    coverage = np.zeros_like(y_true)
    for begin, end in intervals:
        coverage[begin:end] += 1
    assert (coverage == 1).all()

    precision_list = []
    recall_list = []
    ap_list = []

    for begin, end in intervals:
        precision, recall, _ = precision_recall_curve(y_true[begin:end], y_scores[begin:end])
        precision, recall = precision[::-1], recall[::-1]
        precision_list.append(precision)
        recall_list.append(recall)
        ap_list.append(average_precision_score(y_true[begin:end], y_scores[begin:end]))

    mean_ap = np.mean(ap_list)
    std_ap = np.std(ap_list, ddof=1)

    # Interpolation step
    mean_recall = np.linspace(0, 1, 1000)
    precision_list_interpolated = list()
    for precision, recall in zip(precision_list, recall_list):
        precision_list_interpolated.append(np.interp(mean_recall, recall, precision).reshape(1, -1))
    precision_interpolated = np.concatenate(precision_list_interpolated, axis=0)
    mean_precision = np.mean(precision_interpolated, axis=0)
    std_precision = np.std(precision_interpolated, axis=0, ddof=1)

    # No skill line
    # precision_wo_skill, recall_wo_skill, _ = precision_recall_curve(y_true, np.ones_like(y_true))
    # precision_wo_skill, recall_wo_skill = precision_wo_skill[::-1], recall_wo_skill[::-1]
    pos_prevalence = y_true.sum() / len(y_true)

    # Plot the mean ROC curve
    ax.plot(mean_recall, mean_precision)
    ax.fill_between(mean_recall, mean_precision - std_precision, mean_precision + std_precision, color='blue',
                    alpha=0.2)
    ax.plot([0, 1], [pos_prevalence, pos_prevalence], linestyle="--")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP = {mean_ap:.2f} ± {std_ap:.2f})")


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, labels: list, normalize: bool, ax):
    conf_matrix = confusion_matrix(y_true, y_pred, labels=labels, normalize="true" if normalize else None)
    if normalize:
        conf_matrix *= 100
        fmt = ".1f"
    else:
        fmt = "d"
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")
