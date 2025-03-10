# Data-Leakage-HDD-Failure-Prediction

This repository is related to the paper "Quantifying Data Leakage in Failure Prediction Tasks".
In the following we explain how to setup the environment to repeat the experiments or selected parts of them.

### Prerequisites

1. Install Python 3.11. We recommend to use a virtual environment (e.g., with conda) to avoid conflicts with other Python projects.
2. Install the requirements listed in `requirements.txt`

```[shell]
pip install -r requirements.txt
```

3. Download the HDD stats data from Backblaze: [Backblaze Hard Drive Test Data](https://www.backblaze.com/cloud-storage/resources/hard-drive-test-data)
   <br> In the paper, the 2014/15 data and 2016 data was used.
4. For the 2014 data, place the downloaded `data_2014.zip` file in `data/data_2014/data_2014.zip` Don't extract the zip
   file as the script will do this for you. (Please note: Not in `src/data/data_2014/...`! Create the data folder in the project root directory.) 
5. For the 2015 data, repeat previous step analogously.
6. The 2016 data is available quarter-wise. So, for the 2016 Q1 data, place the downloaded `data_Q1_2016.zip` file
   in `data/data_Q1_2016/data_Q1_2016.zip`. Don't extract the zip file as the script will do this for you.
7. Repeat the previous step for the remaining quarters of 2016 analogously.
8. Open `src/config/constants.py` and adjust the Paths or Constants, if desired.

---

### Intermediate results

The required amount of time to run all experiments can be quite long (multiple days on the hardware described in the
paper). Therefore, we provide intermediate results for the hyperparameter tuning, training and independent testing
steps. You can download them from the following link: [Intermediate results download](https://figshare.com/articles/dataset/Intermediate_results_for_the_ICPE_2025_research_paper_artifact_related_to_Quantifying_Data_Leakage_in_Failure_Prediction_Tasks_/28218647)

---

### Argument Parsing for `main.py`

The `main.py` script supports flexible configuration through command-line arguments, enabling reproducibility of
experiments described in the paper. You can also choose to skip parts of the experiments (e.g., hyperparameter tuning)
and enter at any other point, using the provided intermediate results (explanation below).

#### General Configuration (Required Arguments)

- **`--models`** (required): Specify one or more models to include in the experiments.
  Possible values: `MLP`, `LSTM`, `RF`, `HGBC`.

- **`--split_strategies`** (required): Define one or more splitting strategies to be evaluated. These are explained in
  detail in the paper.  
  Possible values:
    - `no-split`: The whole dataset is used for the training, validation and test dataset.
    - `temporal`: Split data temporally.
    - `group-based`: Split data based on predefined groups (here: HDD serial numbers).
    - `random0`: Random splitting of the dataset.

- **`--random_seeds`** (required): Provide one or more random seed values to ensure reproducibility across different
  runs.

- **`--test_year`** (required): The year of the Backblaze HDD Dataset that should be used for the independent test.

#### Experiment Types

- **`--hyperparam_tuning`**: (Optional) Run hyperparameter tuning experiments with one specified random seed.
  Hyperparameter tuning will **not** use the random seeds specified above.
  Example: `--hyperparam_tuning 42`.

- **`--training`**: (Optional) Flag to run training experiments. Assumes that hyperparameters have been tuned before.
  Usage: Add this flag to execute training for multiple random seeds. The results will be aggregated automatically in
  the end.

- **`--independent_test`**: (Optional) Flag to run experiments using an independent test set from the year set above.
  Assumes that training has been completed.
  Usage: Add this flag to execute the independent test for multiple random seeds. The results will be aggregated
  automatically in the end.

#### Evaluation Procedures

- **`--generate_tables_figures`**: (Optional) Generate remaining latex tables and figures for the paper and save them to
  the directories specified in `src/config/constants.py`.
  Usage: Add this flag to generate tables and figures.
- **`--compute_leakage`**: (Optional) Compute the data leakage values with the measure described in the paper for the
  specified split strategies for alpha=0 and alpha=1. Will be saved as `leakage_values.csv` to the tables directory
  specified in `src/config/constants.py`.
  Usage: Add this flag to compute the leakage values.
- **`--aggregate_versions_training`**: (Optional) Aggregate results across multiple seeds. Assumes that training has
  been completed. Will be included automatically if `--training` is specified.  
  Usage: Add this flag to combine outcomes from different random seeds.
- **`--aggregate_versions_independent_test`**: (Optional) Aggregate results across multiple seeds. Assumes that
  independent_test has been completed. Will be included automatically if `--independent_test` is specified.  
  Usage: Add this flag to combine outcomes from different random seeds.

#### Example Usage for ML/DL Experiments

For the deep learning experiments, by default, cuda is used for GPU acceleration, if available.
We strongly recommend using a GPU for the experiments, otherwise the training will take a unreasonably long time.
However, the evaluation procedures (see below) can also be done on a CPU in a reasonable time when using the provided intermediate results.
You can check whether cuda is available by running the following command:

```bash
python -c "import torch; print(torch.cuda.is_available())"
```

Run all paper experiments from scratch. No intermediate results necessary.

```bash
python main.py --models MLP LSTM RF HGBC --split_strategies no-split temporal group-based random0 --random_seeds 43 44 45 46 47 48 49 50 51 52 --test_year 2016 --hyperparam_tuning 42 --training --independent_test
```

Skip the hyperparameter tuning step and start with training.
Prerequisite: You either run the hyperparameter tuning step before or extract `hyperparam_tuning_results.zip` to the
logs directory specified in `src/config/constants.py`.

```bash
python main.py --models MLP LSTM RF HGBC --split_strategies no-split temporal group-based random0 --random_seeds 43 44 45 46 47 48 49 50 51 52 --test_year 2016 --training --independent_test
```

Skip the hyperparameter tuning and training and only predict on the independent test set with the trained models.
Prerequisite: You either run the training step before or extract `training_results.zip` to the logs directory specified
in `src/config/constants.py`.

```bash
python main.py --models MLP LSTM RF HGBC --split_strategies no-split temporal group-based random0 --random_seeds 43 44 45 46 47 48 49 50 51 52 --test_year 2016 --independent_test
```

#### Example Usage for Evaluation Procedures

Generate the results tables and figures that are shown in the paper.
Prerequisite: `training` and `independent_test` have been executed before, or you have extracted `training_results.zip`
and `independent_test_results.zip` to the logs directory specified in `src/config/constants.py`.

```bash
python main.py --models MLP LSTM RF HGBC --split_strategies no-split temporal group-based random0 --random_seeds 43 44 45 46 47 48 49 50 51 52 --test_year 2016 --generate_tables_figures
```

The leakage values for the given split strategies will be computed and saved as `leakage_values.csv` to the tables
directory specified in `src/config/constants.py`.

```bash
python main.py --models MLP LSTM RF HGBC --split_strategies no-split temporal group-based random0 --random_seeds 43 44 45 46 47 48 49 50 51 52 --test_year 2016 --compute_leakage
```

For convenience, these evaluation procedures are also included in the `run_evaluation.sh` script. You can execute it as
follows:

```bash
bash run_evaluation.sh
```

---

### Logging

By default, when running the experiments, the logs are saved in the `logs` directory. The logs will have the same
structure as in the intermediate results provided at the link above.
If desired, the results can be logged to a Neptune Logger in parallel to monitor the experiment status and metrics at
any time in the Neptune Web interface. To do so, you need to provide your Neptune API token and project name in
the `src/config/constants.py` file, and set `USE_NEPTUNE_LOGGER = True`. The logger will then automatically log the
results to your Neptune project.

See: [Neptune.ai Documentation](https://docs.neptune.ai/)

**Please note that for the hyperparameter tuning experiment, the usage of the Neptune Logger will lead to
an `NeptuneFieldCountLimitExceedException` since more than 9000 fields are required for the logs. So only use the
Neptune Logger for the other two experiment types (training and independent test).**

--- 
