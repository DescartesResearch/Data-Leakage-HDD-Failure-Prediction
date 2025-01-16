import random
import warnings
import datetime
from collections import defaultdict
from typing import Optional

import lightning.pytorch as pl
import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, MinMaxScaler, QuantileTransformer
from torch.utils.data import DataLoader, TensorDataset

from data.data_processing import DataUtils, DataFrameProcessor, ProcessingPipeline
from utils.eval import data_leakage
from utils.misc import flatten_dict
from utils.scalers import LogScaler


def get_scaler(scaler_name: str, kwargs: Optional[dict] = None):
    """
    Returns a scaler object based on the provided name, initialized with the provided kwargs.

    :param scaler_name: The name of the scaler, e.g., "StandardScaler"
    :param kwargs: The kwargs to be passed to the scaler (optional)
    :return: The initialized scaler object
    """
    if kwargs is None:
        kwargs = {}
    if scaler_name == "StandardScaler":
        return StandardScaler(**kwargs)
    elif scaler_name == "MinMaxScaler":
        return MinMaxScaler(**kwargs)
    elif scaler_name == "QuantileTransformer":
        return QuantileTransformer(**kwargs)
    elif scaler_name == "LogScaler":
        return LogScaler(**kwargs)
    else:
        raise ValueError(f"Scaler {scaler_name} not found")


class HDDDataModule(pl.LightningDataModule):
    """
    LightningDataModule for the Backblaze HDD dataset.
    """

    def __init__(self, config: dict, num_workers: int = 4):
        """
        :param config: The configuration dictionary. Must contain at least the following keys:
            - task: str | list[str] The task(s) to be performed. Possible values: "binary", "multiclass", "regression".
            - scaler: str | dict The scaler config to be used for scaling the data.
            - hdd_models: str | list[str] The HDD models to be used for the experiment.
            - split_strategy: str | dict The split strategy config to be used for splitting the data.
            - features: str | list[str] The SMART features to be used for the experiment.
            - time_periods: list[tuple|int] The time periods to be used for the experiment.
            - class_intervals: list[int]|tuple[int] The class intervals to be used for the experiment.
        :param num_workers: The number of workers to be used for the dataloaders.
        """
        super().__init__()
        if not isinstance(config, dict):
            raise ValueError(f"Config {config} of type {type(config)} not valid.")
        self.num_workers = num_workers
        self._check_keys_missing(config)
        self.config = config
        self.config["task"] = [self.config["task"]] if isinstance(self.config["task"], str) else self.config["task"]
        self._validate_config()
        self.scaler_y = None
        self.dataset_loaded = False  # The dataset will be loaded in the setup method

    def setup(self, stage: Optional[str] = None) -> None:
        """
        Sets up the data for the respective stage (fit, test, predict).
        :param stage: The stage for which the data should be set up (fit, test, predict). If None, all data is set up.
        :return: None
        """
        if not self.dataset_loaded:
            self._load_data()

        if stage == "fit" or stage is None:
            for ds_typ in ["train", "val", "test"]:
                if hasattr(self, f"X_{ds_typ}"):
                    setattr(self, f"{ds_typ}_dataset",
                            TensorDataset(torch.tensor(getattr(self, f"X_{ds_typ}"), dtype=torch.float32),
                                          *self._ys_to_tensors(getattr(self, f"y_{ds_typ}"))))
        elif stage == "test" or stage == "predict":
            self.test_dataset = TensorDataset(torch.tensor(self.X_test, dtype=torch.float32),
                                              *self._ys_to_tensors(self.y_test))
        else:
            raise ValueError(f"Stage {stage} not implemented in HDDDataModule setup method.")

    def train_dataloader(self, shuffle: bool = True) -> DataLoader:
        if not hasattr(self, "train_dataset") or self.train_dataset is None:
            raise ValueError("Train dataset not loaded yet. Call setup() first.")
        return DataLoader(
            self.train_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.num_workers,
            shuffle=shuffle,
            drop_last=False,
            persistent_workers=True,
        )

    def val_dataloader(self, shuffle: bool = False) -> DataLoader:
        if not hasattr(self, "val_dataset") or self.val_dataset is None:
            raise ValueError("Validation dataset not loaded yet. Call setup() first.")
        return DataLoader(
            self.val_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=True,
        )

    def test_dataloader(self, shuffle: bool = False) -> DataLoader:
        if not hasattr(self, "test_dataset") or self.test_dataset is None:
            raise ValueError("Test dataset not loaded yet. Call setup() first.")
        return DataLoader(
            self.test_dataset,
            batch_size=self.config["batch_size"],
            num_workers=self.num_workers,
            shuffle=shuffle,
            persistent_workers=True,
        )

    def predict_dataloader(self, shuffle: bool = False) -> DataLoader:
        """
        Returns the test dataloader.
        """
        return self.test_dataloader(shuffle=shuffle)

    def get_input_size(self) -> int:
        if not self.dataset_loaded:
            raise RuntimeError("Dataset not loaded yet. Call setup() first.")
        return self.X_train.shape[-1]

    def get_objects(self) -> dict:
        """
        Returns the objects that should be saved to disk after training.
        Currently, only the scalers are supported.
        :return: A dictionary with the objects that should be saved.
        """
        if "save_objects" not in self.config:
            return dict()
        if not isinstance(self.config["save_objects"], list):
            save_objects = [self.config["save_objects"]]
        else:
            save_objects = self.config["save_objects"]

        result = dict()
        for obj_name in save_objects:
            if obj_name == "scaler":
                result[obj_name] = self.scaler
            elif obj_name == "scaler_y":
                result[obj_name] = self.scaler_y
            else:
                raise NotImplementedError(f"HDDDataModule cannot save {obj_name} by now.")
        return result

    @staticmethod
    def get_leakage_value(datamodule: "HDDDataModule",
                          alphas: float | list[float]
                          ) -> dict:
        """
        Calculates the data leakage for the given data module and alphas. For details on the calculation, see the
        utils.eval.data_leakage function.
        :param datamodule: The HDDDataModule object.
        :param alphas: The alpha values to be used for the data leakage calculation.
        :return: A dictionary with the data leakage values for each alpha.
        """
        datamodule.setup("fit")
        train_data = pd.DataFrame(data=datamodule.dates_and_serials["train"], columns=["date", "serial_number"])
        val_data = pd.DataFrame(data=datamodule.dates_and_serials["val"], columns=["date", "serial_number"])
        test_data = pd.DataFrame(data=datamodule.dates_and_serials["test"], columns=["date", "serial_number"])

        if datamodule.config["split_strategy"].get("no_split", False):
            train_val_data = train_data
        else:
            train_val_data = pd.concat([train_data, val_data], axis=0)

        res = {}
        for alpha in alphas:
            res[alpha] = data_leakage(test_data=test_data,
                                      train_data=train_val_data,
                                      group_col="serial_number",
                                      datetime_col="date",
                                      alpha=alpha)
        return res

    @staticmethod
    def _check_keys_missing(config: dict) -> None:
        """
        Validates the config dictionary by checking if all required keys are present.
        Raises a ValueError if a key is missing.

        :param config: The config dictionary
        :return: None
        """
        required_config_keys = ["task", "scaler", "hdd_models", "split_strategy",
                                "features", "time_periods", "class_intervals"]
        for key in required_config_keys:
            if key not in config:
                raise ValueError(f"Key {key} missing in config.")

    def _validate_config(self) -> None:
        """
        Validates the config dictionary by checking if the number of classes (resulting from class_intervals
        and additional_inf_class) is in line with the task.
        For example, for binary classification, there must be exactly two classes.
        Raises a ValueError if the config is not valid.
        :return: None
        """
        if "multiclass" in self.config["task"]:
            if len(self.config["class_intervals"]) == 1 and not self.config.get("additional_inf_class", False):
                raise ValueError("As additional_inf_class==False: Multiclass task requires at least two values in"
                                 "class_intervals to obtain >=3 classes.")
            elif len(self.config["class_intervals"]) == 0:
                raise ValueError("Multiclass task requires >=3 classes, but no class intervals provided.")
        elif "binary" in self.config["task"]:
            if len(self.config["class_intervals"]) == 0 and not self.config.get("additional_inf_class",
                                                                                False):
                raise ValueError("As additional_inf_class==False: Binary task requires exactly one value"
                                 "in class_intervals to obtain 2 classes.")
            elif len(self.config["class_intervals"]) == 1 and self.config.get("additional_inf_class", False):
                raise ValueError("As additional_inf_class==True: Class intervals must be empty.")

    def _load_data(self) -> None:
        """
        Loads the data from disk and applies the preprocessing pipeline.
        The following steps are involved:\n
        0) Check if the dataset is already cached. If so, load it and return. The remaining steps are skipped.
        1) If a scaler from previous runs is provided, exactly the features covered by the scaler are used.
        2) Define Preprocessing Pipeline consisting of the following steps
            - Drop duplicates
            - Discard disks without failures (if specified)
            - Add Remaining Useful Life (RUL) to the dataframe
            - Filter columns (only keep the specified features and the RUL, serial_number and date columns)
            - Drop NaNs and columns with zero variance
        3) Load data from disk for selected time periods, sorted by serial_number and date.
        4) If specified, insert gaps of T_gap days at the split dates and drop the data that lies in the gap.
        5) Apply the preprocessing pipeline as defined above.
        6) Replace the actual SMART values with random values, if requested (for debugging purposes).
        7) Obtain the split indices for train/test/val sets according to the split strategy.
        8) Scale X.
        9) Remember the dates and serial numbers because these will be discarded in the next step.
        10) Add lags and convert to sequential data (3-dim) (if specified)
        11) Remove NaNs that might have been introduced by the lagging; get y_class (labels); save y_counts.
        12) Prepare y for each task scpecified in self.config["task"]. Each list element corresponds to one task.
        13) Cache the dataset.
        :return: None - the loaded data can now be accessed by the setup and get_dataloader methods.
        """

        # PREREQUISITES ###############################################################################################
        # Get the git hash and the config hash for caching
        git_hash = self.config.pop("git_hash") if "git_hash" in self.config.keys() else DataUtils.get_git_hash()
        config_hash = DataUtils.get_config_hash(self.config)

        # Check if the split strategy is set to "test_only" and set the flag accordingly.
        # In this case, the data will not be split. All data (as specified) will be used for testing.
        if self.config["split_strategy"] == "test_only":
            test_only = True
            self.config["split_strategy"] = {}
        else:
            test_only = False
        ds_typs = ["test"] if test_only else ["train", "val", "test"]

        # STEP 0: Check if the dataset is already cached. If so, load it and return. ##################################
        # Otherwise, start preprocessing from scratch.
        if git_hash is not None:
            cached_data = DataUtils.load_dataset(config_hash, git_hash)
            if cached_data is not None:
                for k, v in cached_data.items():
                    setattr(self, k, v)
                self.dataset_loaded = True
                return
        else:
            warnings.warn("Couldn't retrieve git hash. Must create dataset from scratch.")
        print(f"Start dataset preprocessing for config_hash={config_hash}")

        ###############################################################################################################

        # STEP 1: If a scaler from previous runs is provided, exactly the features covered by the scaler are used. ##
        # There might be already loaded objects, e.g., scalers from previous runs
        # that should be used for the current run. Write them into a dictionary.
        self.loaded_objects = self.config.get("loaded_objects", dict())

        if "scaler" in self.loaded_objects:
            # if there is a scaler provided, e.g., from another run, all corresponding features should be used
            scaler_cols = set()
            for sc, cols in self.loaded_objects["scaler"].items():
                if not scaler_cols.isdisjoint(cols):
                    raise ValueError("There must be exactly one scaler for each feature, but for at least one of these"
                                     "features, there are >1 scalers: ", cols)
                scaler_cols.update(cols)
            # Only use the features that are covered by the loaded scalers
            self.config["features"] = scaler_cols

        # STEP 2: Build the preprocessing pipeline ####################################################################
        preproc_pipeline = ProcessingPipeline()
        preproc_pipeline.apply(DataFrameProcessor.drop_duplicates())
        if self.config.get("discard_disks_wo_failure", False):
            preproc_pipeline.apply(DataFrameProcessor.discard_disks_wo_failure())

        preproc_pipeline.apply(
            DataFrameProcessor.add_rul(
                max_rul_threshold=self.config.get("max_rul_threshold", None),
                cap_value=self.config.get("rul_cap_value", None),
                inf_sample_size=self.config.get("inf_sample_size", 1.0),
            )
        ).apply(
            DataFrameProcessor.filter_columns(
                DataUtils.get_smart_features_list(self.config["features"]) + ["rul", "serial_number", "date"])
        )

        if "scaler" in self.loaded_objects:
            preproc_pipeline.apply(DataFrameProcessor.drop_na_rows())
        else:
            preproc_pipeline.apply(
                DataFrameProcessor.drop_na_columns(p_thresh=0.8)
            ).apply(
                DataFrameProcessor.drop_na_rows()
            ).apply(
                DataFrameProcessor.remove_zero_variance_smart_columns()
            )

        # STEP 3: Load data from disk and sort by serial_number and date ##############################################
        data = DataUtils.load_data(
            time_periods=self.config["time_periods"],
            models=self.config["hdd_models"],
            serial_number_filter=self.config.get("serial_number_filter", None)
        )
        data = data.sort_values(by=["serial_number", "date"]).reset_index(drop=True)

        # STEP 4: Insert gaps at the split dates and drop the data that lies in the gap ##############################
        # If temporal_split_dates are set, convert them to datetime objects (if necessary)
        if self.config["split_strategy"].get("temporal_split_dates", None) is not None:
            if not (isinstance(self.config["split_strategy"]["temporal_split_dates"], tuple)
                    and len(self.config["split_strategy"]["temporal_split_dates"]) == 2):
                raise ValueError("temporal_split_dates must be a tuple of two dates in the string format 'yyyy-mm-dd' "
                                 "or two datetime objects.")
            self.config["split_strategy"]["temporal_split_dates"] = tuple(
                datetime.datetime.strptime(d, "%Y-%m-%d") if isinstance(d, str) else d for d in
                self.config["split_strategy"]["temporal_split_dates"]
            )

            # If T_gap is set, insert a gap of T_gap days at the split dates
            # Drop the data that lies in the desired gap between train and test data
            if self.config["split_strategy"].get("T_gap", None) is not None:
                T_gap = self.config["split_strategy"]["T_gap"]
                # parse the date in the format "yyyy-mm-dd" in self.config["temporal_split_date"]
                for split_date in self.config["split_strategy"]["temporal_split_dates"]:
                    # get the date T_gap days before the split date
                    split_date_gap_start = split_date - datetime.timedelta(days=np.ceil(T_gap / 2))
                    split_date_gap_end = split_date + datetime.timedelta(days=np.floor(T_gap / 2))
                    # drop the data that lies in the gap
                    data = data[
                        (data["date"] < split_date_gap_start) | (data["date"] >= split_date_gap_end)].reset_index(
                        drop=True)

        # STEP 5: Apply the preprocessing pipeline that was built above
        X = preproc_pipeline(data).reset_index(drop=True)

        # STEP 6: Replace the actual SMART values with random values, if requested (for debugging purposes) ###########
        if self.config.get("replace_smart_random", False):
            smart_cols = DataUtils.get_smart_cols(X)
            X.loc[:, smart_cols] = np.random.rand(X.shape[0], len(smart_cols))

        # STEP 7: Obtain the split indices for train/test/val sets according to the split strategy ####################
        if test_only:
            self.split_indices = {"test": X.index.tolist()}
        else:
            self.split_indices = self._get_splits(
                X=X,
                group_col="serial_number",
                datetime_col="date",
                **self.config["split_strategy"],
            )

        # Sanity check that serial numbers are not spread across train, val and test
        # Skip the check if overlap_HDD_ratio is set to 0 or if no_split is set to True
        if self.config["split_strategy"].get("overlap_HDD_ratio", .0) == .0 and not self.config["split_strategy"].get(
                "no_split", False) and not test_only:
            for i, ds_typ in enumerate(["train", "val", "test"]):
                serial_numbers0 = set(X.iloc[self.split_indices[ds_typ]]["serial_number"])
                for j, ds_typ2 in enumerate(["train", "val", "test"]):
                    if not j > i:
                        continue
                    serial_numbers1 = set(X.iloc[self.split_indices[ds_typ2]]["serial_number"])
                    assert len(serial_numbers0.intersection(
                        serial_numbers1)) == 0, f"Serial numbers are spread across split {i} and {j}."

        # Step 8: Scale X #############################################################################################
        if self.config.get("scaler", None) is not None:
            # First, get scaler(s) from loaded objects or create new one(s)
            if "scaler" in self.loaded_objects:
                print("Got scaler from loaded objects!")
                self.scaler = self.loaded_objects["scaler"]
            elif isinstance(self.config["scaler"], dict):
                assert not test_only, "Scalers must be the same for train and test data, so scaler must be retrieved from loaded objects."
                self.scaler = {
                    get_scaler(sc_name): X.columns.intersection(DataUtils.get_smart_features_list(features))
                    for sc_name, features in self.config["scaler"].items()}
            else:
                assert not test_only, "Scalers must be the same for train and test data, so scaler must be retrieved from loaded objects."
                self.scaler = {get_scaler(self.config["scaler"]): X.columns.intersection(
                    DataUtils.get_smart_features_list("all"))}

            # Sanity check: Each column has exactly one scaler
            all_cols = set()
            for cols in self.scaler.values():
                if not all_cols.isdisjoint(cols):
                    raise ValueError("There must be exactly one scaler for each feature, but for at least one of these"
                                     "features, there are >1 scalers: ", cols)
                all_cols.update(cols)
            if all_cols != set(DataUtils.get_smart_cols(X)):
                raise ValueError(
                    "All columns must be covered by the scalers, but the following columns are not covered: ",
                    set(DataUtils.get_smart_cols(X)) - all_cols)

            # Apply the scaling
            for sc, cols in self.scaler.items():
                if test_only:
                    X.iloc[self.split_indices["test"]] = DataFrameProcessor.scale(scaler=sc, cols=cols)(X)
                else:
                    X.iloc[self.split_indices["train"]], X.iloc[self.split_indices["val"]], X.iloc[
                        self.split_indices["test"]] = ProcessingPipeline().apply_to_all(
                        DataFrameProcessor.scale(scaler=sc, cols=cols))(X.iloc[self.split_indices["train"]],
                                                                        X.iloc[self.split_indices["val"]],
                                                                        X.iloc[self.split_indices["test"]])

        # STEP 9: Remember the dates and serial numbers because these will be discarded in the next step ##############
        self.dates_and_serials = dict()
        for ds_typ in ds_typs:
            self.dates_and_serials[ds_typ] = X.loc[:, ["date", "serial_number"]].values
            self.dates_and_serials[ds_typ] = self.dates_and_serials[ds_typ][self.split_indices[ds_typ]]

        # STEP 10: Add lags and convert to sequential data (3-dim), if specified #######################################
        if self.config.get("sequential", False) or self.config.get("lags", None) is not None:
            X = DataFrameProcessor.equidistant_daterange_imputation()(X)
            y_rul = X["rul"].values

            # Sanity check: Check that for each serial number, there are no gaps in the dates and the values are sorted by date.
            # Function to validate sorting:
            assert X.groupby('serial_number', observed=True)['date'].apply(
                lambda x: x.is_monotonic_increasing).all()
            # Calculate the difference in days between consecutive dates for each 'serial_number'
            date_diff = X.groupby('serial_number', observed=True)['date'].diff().dt.days
            # Find serial numbers with gaps of at least 2 days
            assert not (date_diff >= 2).any()

            # Add lags and convert to sequential data (3D numpy array) if necessary
            if self.config.get("sequential", False):
                assert self.config.get("lags", None) is not None, "Lags must be specified for sequential data."
                X = DataUtils.df_to_series(X, lags=self.config["lags"])
            else:
                X = DataFrameProcessor.add_lags_to_smart_cols(self.config["lags"])(X)
        else:
            y_rul = X["rul"].values

        if isinstance(X, pd.DataFrame):
            X = DataFrameProcessor.only_keep_smart_cols()(X)
            X = X.values

        # NOTE: From here on X and y_rul are numpy arrays instead of pandas dataframes!

        # STEP 11 ######################################################################################################

        # Set attributes X_train, X_val, X_test, y_rul_train, y_rul_val, y_rul_test
        for ds_typ in ds_typs:
            setattr(self, f"X_{ds_typ}", X[self.split_indices[ds_typ]])
            setattr(self, f"y_rul_{ds_typ}", y_rul[self.split_indices[ds_typ]])

        # Make sure that no NaN values are contained in the data by removing rows with NaN values
        # Get rows with NaN values
        nan_rows = dict()
        for ds_typ in ds_typs:
            nan_rows[ds_typ] = np.isnan(getattr(self, f"X_{ds_typ}")).any(
                axis=tuple(range(1, getattr(self, f"X_{ds_typ}").ndim)))

        # Only keep rows without NaN values for both X and y
        for ds_typ in ds_typs:
            self.dates_and_serials[ds_typ] = self.dates_and_serials[ds_typ][~nan_rows[ds_typ]]
            setattr(self, f"X_{ds_typ}", getattr(self, f"X_{ds_typ}")[~nan_rows[ds_typ]])
            setattr(self, f"y_rul_{ds_typ}", getattr(self, f"y_rul_{ds_typ}")[~nan_rows[ds_typ]])
            setattr(self, f"y_class_{ds_typ}", DataUtils.get_y_class(getattr(self, f"y_rul_{ds_typ}"),
                                                                     class_intervals=self.config["class_intervals"],
                                                                     additional_inf_class=self.config.get(
                                                                         "additional_inf_class", False)))

        self.y_counts = {}
        for y_typ in ["rul", "class"]:
            self.y_counts[y_typ] = {ds_typ: self._get_y_counts(getattr(self, f"y_{y_typ}_{ds_typ}"))
                                    for ds_typ in ds_typs}

        # STEP 12 ######################################################################################################
        # Prepare y for each task. Each list element corresponds to one task.
        # For regression, optionally apply a scaler.
        for ds_typ in ds_typs:
            setattr(self, f"y_{ds_typ}", list())  # prepare list for each task
        for task in self.config["task"]:
            task_y = dict()
            for ds_typ in ds_typs:
                if task == "binary":
                    # assert that maximum label for all train, val and test data is 1
                    assert np.max(getattr(self,
                                          f"y_class_{ds_typ}")) <= 1, "Binary classification requires only labels 0 and 1."
                    task_y[ds_typ] = getattr(self, f"y_class_{ds_typ}")
                elif task == "multiclass":
                    assert np.max(getattr(self,
                                          f"y_class_{ds_typ}")) > 1, "Multiclass classification requires more than two labels."
                    task_y[ds_typ] = getattr(self, f"y_class_{ds_typ}")
                elif task == "encoding":
                    task_y[ds_typ] = getattr(self, f"X_{ds_typ}")
                elif task == "regression":
                    if test_only:
                        raise NotImplementedError(
                            "Regression task not implemented for test_only mode. Requires loading the max value from original training dataset.")
                    # set np.inf to the max value + 14
                    if not hasattr(self, "regression_inf_replace_value"):
                        self.regression_inf_replace_value = np.max(self.y_rul_train[self.y_rul_train != np.inf]) + 14
                    task_y[ds_typ] = np.nan_to_num(getattr(self, f"y_rul_{ds_typ}"),
                                                   posinf=self.regression_inf_replace_value)
                    if self.config.get("scaler_y", None) is not None:  # Scaling of y demanded
                        if not hasattr(self, "scaler_y"):
                            if "scaler_y" in self.loaded_objects:
                                self.scaler_y = self.loaded_objects["scaler_y"]
                                print("Got fitted scaler_y from loaded objects!")
                            else:
                                self.scaler_y = get_scaler(self.config["scaler_y"])
                                self.scaler_y.fit(task_y["train"].reshape(-1, 1))
                        task_y[ds_typ] = self.scaler_y.transform(task_y[ds_typ].reshape(-1, 1))
                else:
                    raise ValueError(f"Task {task} not implemented.")

            for ds_typ in ds_typs:
                getattr(self, f"y_{ds_typ}").append(task_y[ds_typ])

        # STEP 13: Cache the dataset ##################################################################################
        if git_hash is not None:
            dataset_dict = {
                "y_counts": self.y_counts,
                "scaler": self.scaler,
                "scaler_y": self.scaler_y,
                "split_indices": self.split_indices,
                "dates_and_serials": self.dates_and_serials
            }
            for ds_typ in ds_typs:
                dataset_dict[f"X_{ds_typ}"] = getattr(self, f"X_{ds_typ}")
                dataset_dict[f"y_{ds_typ}"] = getattr(self, f"y_{ds_typ}")
            DataUtils.save_dataset(dataset_dict, config_hash, git_hash)
        self.dataset_loaded = True

    @staticmethod
    def _get_splits(
            X: pd.DataFrame,
            group_col: str = "serial_number",
            datetime_col: str = "date",
            no_split: bool = False,
            overlap_HDD_ratio: float = .0,
            temporal_split_dates: tuple[datetime.date, datetime.date] | None = None,
            temporal_split_ratio: float = 1.,
            temporal_split_of_non_overlapping_hdds: bool = False,
            val_ratio: float = .1,
            test_ratio: float = .3,
    ) -> dict:
        """
        Splits the data into train, val and test sets according to the specified split strategy.

        :param X: The dataframe to split.
        :param group_col: The column that specifies the groups (e.g., serial numbers).
        :param datetime_col: The column that contains the datetime information.
        :param no_split: Set to True if no split should be performed. The remaining parameters will be ignored. The function will return all indices for train, val and test.
        :param overlap_HDD_ratio: The ratio of HDDs that are in train, val and test set.
        :param temporal_split_dates: The dates to split the data into train and test and the train set further into
         train and val set.
        :param temporal_split_ratio: The ratio of HDDs that are split temporally according to the split dates.
        :param temporal_split_of_non_overlapping_hdds: Set to True if non-overlapping HDDs should be split temporally.
        :param val_ratio: The ratio of HDDs that are in the validation set.
        :param test_ratio: The ratio of HDDs that are in the test set.
        """

        split_indices = defaultdict(list)  # result dict

        if no_split:
            # Simply return all indices for train, val and test
            for ds_typ in ["train", "val", "test"]:
                split_indices[ds_typ].extend(X.index)
            return split_indices

        # Randomly sample overlap_HDD_ratio of the HDDs (defined by the group_col)
        hdds = {k0: {k1: list()} for k0 in ["overlapping", "non_overlapping"] for k1 in
                ["temporal_split", "no_temporal_split"]}
        all_HDDs = X[group_col].unique().tolist()
        overlapping_HDDs = random.sample(all_HDDs, int(len(all_HDDs) * overlap_HDD_ratio))
        # Randomly sample temporal_split_ratio of the overlapping_HDDs
        hdds["overlapping"]["temporal_split"] = random.sample(tuple(overlapping_HDDs),
                                                              int(len(overlapping_HDDs) * temporal_split_ratio))
        hdds["overlapping"]["no_temporal_split"] = list(
            set(overlapping_HDDs) - set(hdds["overlapping"]["temporal_split"]))

        non_overlapping_HDDs = list(set(all_HDDs) - set(overlapping_HDDs))
        if temporal_split_of_non_overlapping_hdds:
            hdds["non_overlapping"]["temporal_split"] = non_overlapping_HDDs
            hdds["non_overlapping"]["no_temporal_split"] = list()
        else:
            hdds["non_overlapping"]["temporal_split"] = list()
            hdds["non_overlapping"]["no_temporal_split"] = non_overlapping_HDDs

        # Sanity check: Each HDD is exactly in one of the four categories
        _all_union = set()
        for k0, hdd_list0 in (fd := flatten_dict(hdds)).items():
            hdd_list0: list
            _all_union.update(hdd_list0)
            for k1, hdd_list1 in fd.items():
                if k0 == k1:
                    continue
                assert set(hdd_list0).isdisjoint(hdd_list1)
        assert _all_union == set(X[group_col])

        # Handle the four cases/categories of HDDs:
        # Case 0: Non_overlapping HDDs (i.e., each HDD in exactly one of train, val and test)
        # Case 0-0: Split non_overlapping HDDs into train/val/test without temporal split according to test_ratio
        # Case 0-1: " with temporal split, then for train/test HDDs only keep data before/after the split_day
        # Case 1: Overlapping HDDs (i.e., HDDs that may appear in train AND test)
        # Case 1-0: Split overlapping HDDs without temporal split according to test_ratio -> Simply a random split of the indices
        # Case 1-1: " with temporal split " -> All data before split_day is train, all data after is test

        for case_id0, overlap in enumerate(["non_overlapping", "overlapping"]):
            for case_id1, temporal in enumerate(["no_temporal_split", "temporal_split"]):
                case = f"{case_id0}-{case_id1}"
                curr_hdds: list[str] = hdds[overlap][temporal]
                if curr_hdds == list():
                    continue
                if case.startswith("0"):
                    # randomly sample test_ratio samples from the set curr_hdds
                    curr_hdds_test = random.sample(curr_hdds, int(len(curr_hdds) * test_ratio))
                    curr_hdds_train_val = list(set(curr_hdds) - set(curr_hdds_test))
                    curr_hdds_val = random.sample(curr_hdds_train_val, int(len(curr_hdds) * val_ratio))
                    curr_hdds_train = list(set(curr_hdds_train_val) - set(curr_hdds_val))
                    # Get the indices of the HDDs in the respective sets
                    for hdd_set, ds_typ in zip([curr_hdds_train, curr_hdds_val, curr_hdds_test],
                                               ["train", "val", "test"]):
                        if case.endswith("0"):
                            curr_indices = X[X[group_col].isin(hdd_set)].index
                        else:
                            if ds_typ == "train":
                                curr_indices = X[
                                    (X[group_col].isin(hdd_set)) & (X[datetime_col] < temporal_split_dates[0])].index
                            elif ds_typ == "val":
                                curr_indices = X[
                                    (X[group_col].isin(hdd_set)) & (X[datetime_col] >= temporal_split_dates[0]) & (
                                            X[datetime_col] < temporal_split_dates[1])].index
                            else:
                                curr_indices = X[
                                    (X[group_col].isin(hdd_set)) & (X[datetime_col] >= temporal_split_dates[1])].index
                        split_indices[ds_typ].extend(curr_indices)
                elif case == "1-0":
                    all_curr_indices = X[X[group_col].isin(curr_hdds)].index.tolist()
                    curr_indices_test = random.sample(all_curr_indices, int(len(all_curr_indices) * test_ratio))
                    curr_indices_train_val = list(set(all_curr_indices) - set(curr_indices_test))
                    curr_indices_val = random.sample(curr_indices_train_val, int(len(all_curr_indices) * val_ratio))
                    curr_indices_train = list(set(curr_indices_train_val) - set(curr_indices_val))
                    for curr_indices, ds_typ in zip([curr_indices_train, curr_indices_val, curr_indices_test],
                                                    ["train", "val", "test"]):
                        split_indices[ds_typ].extend(curr_indices)
                elif case == "1-1":
                    split_indices["train"].extend(
                        X[(X[group_col].isin(curr_hdds)) & (X[datetime_col] < temporal_split_dates[0])].index)
                    split_indices["val"].extend(X[(X[group_col].isin(curr_hdds)) & (
                            X[datetime_col] >= temporal_split_dates[0]) & (
                                                          X[datetime_col] < temporal_split_dates[1])].index)
                    split_indices["test"].extend(
                        X[(X[group_col].isin(curr_hdds)) & (X[datetime_col] >= temporal_split_dates[1])].index)

        assert set(split_indices["train"]).union(split_indices["val"]).union(split_indices["test"]) == set(
            X.index), "Not all original indices are used in the splits!"

        # Sanity check: Check that all indices are disjoint to avoid data leakage
        assert set(split_indices["train"]).isdisjoint(split_indices["val"]), "Train and val indices are not disjoint."
        assert set(split_indices["train"]).isdisjoint(split_indices["test"]), "Train and test indices are not disjoint."
        assert set(split_indices["val"]).isdisjoint(split_indices["test"]), "Val and test indices are not disjoint."

        return split_indices

    def _get_y_counts(self, y_train) -> dict:
        """
        Get the counts of the unique values in y_train.
        :param y_train: The y-values.
        :return: A dictionary with keys absolute and relative with the absolute/relative counts of the unique values.
        """
        y_counts = dict()
        y_counts["absolute"] = pd.Series(y_train.flatten()).value_counts(normalize=False).sort_index()
        y_counts["relative"] = pd.Series(y_train.flatten()).value_counts(normalize=True).sort_index()
        return y_counts

    def _ys_to_tensors(self, y: list) -> list[torch.Tensor]:
        """
        Converts the y-values to torch tensors.
        :param y: A list of y, each for a different task.
        :return: A list of torch tensors.
        """
        tensors = []
        for y_i, task_i in zip(y, self.config["task"]):
            if task_i == "multiclass":
                tensors.append(torch.tensor(y_i, dtype=torch.long).flatten())
            elif task_i == "binary":
                tensors.append(torch.tensor(y_i, dtype=torch.float32).reshape(-1, 1))
            elif task_i in ["regression", "encoding"]:
                tensors.append(torch.tensor(y_i, dtype=torch.float32))
            else:
                raise ValueError(f"Task {task_i} not implemented.")
        return tensors
