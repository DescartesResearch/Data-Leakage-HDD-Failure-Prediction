import datetime
import pickle
from collections import defaultdict
from typing import Optional

import neptune
import numpy as np
import pandas as pd
import atexit
import shutil
from pathlib import Path

from config.constants import Constants, Paths
from utils.eval import get_metrics_and_figures, aggregate_iterations
from utils.misc import rec_defaultdict, flatten_dict, unflatten_dict


class NeptuneUtils:
    """
    Collection of utility functions to interact with the Neptune API.
    """

    @staticmethod
    def get_run_by_id(run_id: str):
        return neptune.Run(with_id=run_id,
                           project=Constants.NEPTUNE_LOGGER_PROJECT,
                           api_token=Constants.NEPTUNE_LOGGER_API_TOKEN,
                           capture_stdout=False,
                           capture_stderr=False,
                           capture_hardware_metrics=False,
                           )

    @staticmethod
    def get_run_id_by_name(name: str, runs_table_df: pd.DataFrame = None) -> Optional[str]:
        """
        The name must be unique in the runs table. Otherwise a RuntimeError is raised. If none of the runs matches the name,
        None is returned.
        :param name: The name of the requested run.
        :param runs_table_df: Optionally provide a runs_table, else the runs_table is loaded by the function.
        :return: The corresponding Neptune run_id, if available, None otherwise.
        """
        if runs_table_df is None:
            runs_table_df = NeptuneUtils.load_runs_table()
        if "sys/name" not in runs_table_df.columns:
            return None
        matches = runs_table_df[runs_table_df["sys/name"] == name].shape[0]
        if matches > 1:
            raise ValueError("Multiple runs with the same name found!")
        elif matches == 0:
            return None
        else:
            return runs_table_df[runs_table_df["sys/name"] == name]["sys/id"].values[0]

    @staticmethod
    def get_run_by_name(name: str, runs_table_df: pd.DataFrame = None) -> Optional[neptune.Run]:
        """
        The name must be unique in the runs table. Otherwise a RuntimeError is raised. If none of the runs matches the name,
        None is returned.
        :param name: The name of the requested run.
        :param runs_table_df: Optionally provide a runs_table, else the runs_table is loaded by the function.
        :return: The corresponding Neptune run, if available, None otherwise.
        """
        if (run_id := NeptuneUtils.get_run_id_by_name(name, runs_table_df)) is not None:
            return NeptuneUtils.get_run_by_id(run_id)
        else:
            return None

    @staticmethod
    def load_runs_table(criteria: dict = None) -> pd.DataFrame:
        """
        Load runs that meet the given criteria.
        :param criteria: (Optional) The criteria to filter runs. The keys are the column names and the values are the values to filter for (equality).
        :return: Neptune runs table of runs that match the criteria.
        """
        neptune_project = neptune.init_project(project=Constants.NEPTUNE_LOGGER_PROJECT,
                                               api_token=Constants.NEPTUNE_LOGGER_API_TOKEN)
        runs_table = neptune_project.fetch_runs_table()
        runs = runs_table.to_pandas()
        if criteria is not None:
            runs = NeptuneUtils.filter_runs(runs, criteria)
        return runs

    @staticmethod
    def filter_runs(runs_table: pd.DataFrame, criteria: dict = None, **further_criteria) -> pd.DataFrame:
        """
        Filters the runs table according to the given criteria.
        :param runs_table: The runs table to filter.
        :param criteria: The criteria to filter the runs table. The keys are the column names and the values are the values to filter for (equality).
        :param further_criteria: Additional criteria to filter the runs table.
        :return: The filtered runs table with only rows that meet the criteria.
        """
        if criteria is None:
            criteria = {}
        criteria.update(further_criteria)
        keys = list(criteria.keys())
        values = list(criteria.values())
        selected_rows = runs_table[keys[0]] == values[0]
        for i in range(1, len(keys)):
            selected_rows &= (runs_table[keys[i]] == values[i])
        runs_table = runs_table[selected_rows].reset_index(drop=True)
        return runs_table

    @staticmethod
    def delete_if_exists(run: neptune.Run, paths: str | list[str]) -> bool:
        """
        Deletes the given paths from the neptune run if it exists and syncs in the end.
        :param run: Neptune Run
        :param paths: The neptune paths, e.g., "fold-0/metrics"
        :return: True if all deleted, False if any not available
        """
        any_missing = False
        if isinstance(paths, str):
            paths = [paths]
        # remove nan values
        paths = [p for p in paths if p is not None]
        assert len(paths) > 0
        for path in paths:
            if run.exists(path):
                del run[path]
            else:
                any_missing = False
        run.sync()
        return not any_missing
