from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import pyperclip
from matplotlib import pyplot as plt
import seaborn as sns

from config.constants import Paths
from training.logging import get_results_df


def generate_latex_table(models: list[str],
                         metrics: list[str],
                         datasets: dict,
                         results: dict | pd.DataFrame,
                         caption: str,
                         label: str,
                         sub_dataset_name_mapping=None,
                         raise_on_missing=True,
                         fill_value: str = "N/A",
                         adjustbox: bool = False,
                         to_clipboard: bool = True) -> str:
    """
    Generates a LaTeX table based on the given models, metrics, datasets, and results.

    :param models: List of model names (e.g., ["LSTM", "SVM"]).
    :param metrics: List of metric names (e.g., ["AUC", "MCC"]).
    :param datasets: Dictionary defining the dataset structure (e.g., {"2014/2015": ["train=test", "temporal", "group-based"]}).
    :param results: Dictionary containing the results (e.g., results["LSTM"]["2014/2015"]["temporal"]["MCC"]).
    :param caption: Caption for the LaTeX table.
    :param label: Label for referencing the table in LaTeX documents.
    :param raise_on_missing: Whether to raise an error if a value is missing.
    :param fill_value: The value to use for missing entries (default is "N/A").
    :param adjustbox: Whether to use the adjustbox package for resizing the table.
    :param to_clipboard: Whether to copy the LaTeX table to the clipboard.
    :return: The LaTeX table as a string. Also copies the LaTeX table to the clipboard, if specified.
    """
    if sub_dataset_name_mapping is None:
        sub_dataset_name_mapping = {}

    # Header row for models
    num_metrics = len(metrics)
    num_columns_per_model = num_metrics * len(models)
    header_row_models = " & ".join([f"\\multicolumn{{{num_metrics}}}{{c}}{{{model}}}" for model in models])

    # Header row for metrics under each model
    header_row_metrics = " & ".join(metrics * len(models))

    # Start building the LaTeX table
    latex_table = [
        "\\begin{table*}",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\begin{adjustbox}{width=\\textwidth}" if adjustbox else "",
        "\\begin{tabular}{l" + "".join(["c" * num_metrics] * len(models)) + "}",
        "\\toprule",
        # "\\hline",
        f" & {header_row_models} \\\\",
        " ".join(
            [f"\\cmidrule(lr){{{2 + len(metrics) * i}-{1 + len(metrics) * (i + 1)}}}" for i in range(len(models))]),
        f"\\multicolumn{{{1}}}{{c}}{{Split}} & {header_row_metrics} \\\\",
        "\\midrule"
    ]

    # Fill in the rows for datasets and their sub-rows
    for i, (main_dataset, sub_datasets) in enumerate(datasets.items()):
        # Multi-row for the main dataset name, rotated 90 degrees
        latex_table.append(f"\\textbf{{{main_dataset}}}" + " & " * (len(models) * len(metrics)) + " \\\\")

        for j, sub_dataset in enumerate(sub_datasets):
            row = [sub_dataset if sub_dataset not in sub_dataset_name_mapping else sub_dataset_name_mapping[
                sub_dataset]]  # Start row with the sub-dataset description
            for model in models:
                for metric in metrics:
                    try:
                        # Get the result for the given model, dataset, sub-dataset, and metric
                        if isinstance(results, dict):
                            value = results[model][main_dataset][sub_dataset].get(metric, fill_value)
                        elif isinstance(results, pd.DataFrame):
                            value = results.loc[(model == results["model"]) & (main_dataset == results["year"]) & (
                                    sub_dataset == results["split"]), metric].values
                            if len(value) == 0:
                                raise KeyError()
                            elif len(value) > 1:
                                raise ValueError("Multiple values found")
                            value = value[0]
                    except KeyError:
                        if raise_on_missing:
                            raise ValueError(f"Missing value for {model}, {main_dataset}, {sub_dataset}, {metric}")
                        value = fill_value
                    row.append(value)
            latex_table.append(" & ".join(row) + " \\\\")

        latex_table.append("\\\\" if i < len(sub_datasets) else "")

    # End the table
    latex_table.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{adjustbox}" if adjustbox else "",
        "\\end{table*}"
    ])

    s = "\n".join(latex_table)

    # Print the LaTeX table to the console
    # print(s)

    if to_clipboard:
        # Copy the LaTeX table to the clipboard
        pyperclip.copy(s)

    return s


def value_and_error_to_latex(mean: float, std: float, precision: int = 2, colorcoding: bool = False) -> str:
    res = "${:.{precision}f} \\pm {:.{precision}f}$".format(mean, std, precision=precision)
    if colorcoding:
        if (mean - std - (0.6 / 10 ** precision)) > 0:
            color = "darkred"
        else:
            color = "darkgreen"
        res = res.replace("$", f"$\\textcolor{{{color}}}{{", 1)
        res = res[:-1] + "}$"
    return res


def generate_tables_and_diagrams(
        model_names: list[str],
        split_strategies: list[str],
        default_year: str = "2014/2015",
        independent_test_year: str = "2016",
        raise_on_missing: bool = True,
        figures_dir: Path = Paths.FIGURES,
        tables_dir: Path = Paths.TABLES,
) -> None:
    """
    Generates tables and diagrams as shown in the paper.
    """
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    exp_names = set()
    for model_name in model_names:
        for split_strategy in split_strategies:
            exp_names.add(f"model={model_name}_split={split_strategy}")
            exp_names.add(f"independent-test_year={independent_test_year}_model={model_name}_split={split_strategy}")
    exp_names = list(exp_names)

    metric_names = ['F1', 'G_mean', 'MCC']
    colname_mapping = {
        'mean/test/multiclass/class_0/MCC': 'MCC mean',
        'std/test/multiclass/class_0/MCC': 'MCC std',
        'mean/test/multiclass/class_0/G_mean': 'G_mean mean',
        'std/test/multiclass/class_0/G_mean': 'G_mean std',
        'mean/test/multiclass/class_0/F1': 'F1 mean',
        'std/test/multiclass/class_0/F1': 'F1 std',
    }
    results_df = get_results_df(exp_names, ds_typs=["test"])
    # Only keep the columns that are required.
    results_df = results_df[["year", "model", "split"] + list(colname_mapping.keys())]
    results_df = results_df.rename(columns=colname_mapping)
    results_df["year"] = results_df["year"].fillna(default_year)

    # Generate Tables:
    datasets = {
        default_year: split_strategies,
        independent_test_year: split_strategies
    }

    for table_type in ["performance_optimism", "results"]:
        if table_type == "performance_optimism":
            curr_results_df = pd.merge(left=results_df[results_df.year == default_year],
                                       right=results_df[results_df.year == independent_test_year],
                                       on=["model", "split"],
                                       suffixes=("_default", "_independent"))
            for metric in metric_names:
                mean_name = f"{metric} mean"
                std_name = f"{metric} std"
                curr_results_df[mean_name] = curr_results_df[f"{mean_name}_default"] - curr_results_df[
                    f"{mean_name}_independent"]
                curr_results_df[std_name] = (curr_results_df[f"{std_name}_default"] ** 2 + curr_results_df[
                    f"{std_name}_independent"] ** 2) ** 0.5
                curr_results_df.drop(columns=[f"{mean_name}_default", f"{std_name}_default", f"{mean_name}_independent",
                                              f"{std_name}_independent"], inplace=True)
            curr_results_df.drop(columns=["year_independent"], inplace=True)
            curr_results_df.rename(columns={"year_default": "year"}, inplace=True)
            curr_datasets = {default_year: datasets[default_year]}
            colorcoding = True
            caption = "Performance Optimism (PO) of different split strategies. Red values indicate an overestimation of the predictive power, green values indicate a realistic or underestimation."
            label = "tab:po_results"
            po_df = curr_results_df.copy(deep=True)  # Copy for generating the figures later
        else:
            curr_datasets = datasets.copy()
            caption = "Performance of models trained and tested with different split strategies applied to the 2014/15 dataset and performance on the held-out test data of scientific interest from 2016."
            label = "tab:results"
            curr_results_df = results_df.copy(deep=True)
            colorcoding = False

        curr_results_df.to_csv(tables_dir / f"{table_type}.csv", index=False)

        for metric in metric_names:
            mean_name = f"{metric} mean"
            std_name = f"{metric} std"
            curr_results_df[metric] = curr_results_df.apply(
                lambda row: value_and_error_to_latex(row[mean_name], row[std_name], colorcoding=colorcoding),
                axis=1)
            curr_results_df.drop(columns=[mean_name, std_name], inplace=True)

        tab_str = generate_latex_table(
            models=model_names,
            metrics=metric_names,
            datasets=curr_datasets,
            results=curr_results_df,
            caption=caption,
            label=label,
            raise_on_missing=raise_on_missing,
            to_clipboard=False
        )

        with open(tables_dir / f"{table_type}.tex", "w") as f:
            f.write(tab_str)

    # Generate paper figures
    for metric in metric_names:
        x = np.arange(len(model_names))
        width = 0.2  # Width of bars

        # get colorblind palette
        colorblind = sns.color_palette("colorblind")

        # Create the figure and axes
        fig, ax = plt.subplots(figsize=(4.8, 3.6))

        # Adjust settings for matplotlib so that it is suitable for a paper

        # Plot bars with error bars for each split strategy
        split_strategy_name_mapping = {
            "random0": "Random",
            "no-split": "No split",
            "temporal": "Temporal",
            "group-based": "Group-based"
        }

        for i, split_strategy in enumerate(split_strategies):
            values = []
            stds = []
            for model in model_names:
                value = po_df[(po_df["split"] == split_strategy) & (po_df["model"] == model)][metric + " mean"].values
                std = po_df[(po_df["split"] == split_strategy) & (po_df["model"] == model)][metric + " std"].values
                if len(value) == 0 or len(std) == 0:
                    raise ValueError(f"Missing value for {split_strategy} in {metric}")
                elif len(value) > 1 or len(std) > 1:
                    raise ValueError(f"Multiple values found for {split_strategy} in {metric}")
                else:
                    values.append(value[0])
                    stds.append(std[0])
            ax.bar(x + i * width,
                   values,
                   width,
                   yerr=stds,
                   label=split_strategy_name_mapping[split_strategy],
                   capsize=5,
                   color=colorblind[i])

        # Labeling and legend
        ax.set_xticks(x + width * 1.5)
        ax.set_xticklabels(model_names)
        # set y axis limits
        ax.set_ylim(-0.2, 1.15)
        # ax.set_xlabel("Model")
        ax.set_ylabel(f"{metric} optimism")
        ax.legend(ncols=2, loc="upper center")
        plt.tight_layout()
        plt.savefig(figures_dir / f"{metric}_optimism.pdf")
        plt.close()
        # plt.savefig(f"../figures/{metric}_optimism.pdf")
        # plt.show()
