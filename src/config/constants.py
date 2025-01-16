from pathlib import Path


class Paths:
    """
    Clean organisation of relevant paths of the project to avoid things like "C:\\Users\\MaxMustermann\\Projects\\..."
    """
    PROJECT = Path(__file__).resolve().parents[2]
    DATA = PROJECT / "data"
    DATA_CACHE = DATA / "cache"
    RAY_RESULTS = PROJECT / "ray_results"
    LOGS = PROJECT / "logs"
    MODELS = PROJECT / "trained_models"
    TMP = PROJECT / "tmp"
    FIGURES = PROJECT / "paper_figures"
    TABLES = PROJECT / "paper_tables"

    @staticmethod
    def get_data_path(year: int, quarter: int, subdir: str = None):
        """
        Get the path of Backblaze HDD data for a specific quarter and year.
        """
        path = Paths.DATA / f"data_Q{quarter}_{year}"
        if subdir is not None:
            path = path / subdir
        return path


class Constants:
    """
    Constants used throughout the project.
    """
    # The model number of the HDD model used in the experiments
    SEAGATE_4TB = "ST4000DM000"
    # Set this to True to log to the File System
    USE_FS_LOGGER = True
    # Neptune AI Logger Project, API Token and prefix of the Run IDs
    USE_NEPTUNE_LOGGER = False
    NEPTUNE_LOGGER_PROJECT = ...
    NEPTUNE_LOGGER_API_TOKEN = ...
    NEPTUNE_LOGGER_MODE = "async"
    NEPTUNE_ASYNC_NO_PROGRESS_THRESHOLD = 3600 * 24
    NEPTUNE_ASYNC_LAG_THRESHOLD = 3600 * 24 * 3
    # Prefixes used as subdirectories for different kinds of data being logged for each run (or folds within runs)
    LOGGING_PREDICTIONS_PREFIX = "predictions"
    LOGGING_THRESHOLD_PREFIX = "thresholds"
    LOGGING_METRICS_PREFIX = "metrics"
    LOGGING_FIGURES_PREFIX = "figures"
    # Default seed if none provided in the specific configs of the experiments
    DEFAULT_SEED = 42
    # Number of workers used for the dataloaders
    NUM_WORKERS = 4
    # If the RAM limit is exceeded in the main.py script, the script will be aborted.
    # Only works on Linux. Set to 0 to disable the limit.
    MEMORY_LIMIT = 80  # in GB
