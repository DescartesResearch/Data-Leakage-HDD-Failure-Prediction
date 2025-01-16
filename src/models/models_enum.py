from enum import Enum


class PytorchModels(Enum):
    MLP = "MLP"
    LSTM = "LSTM"


class SKLearnModels(Enum):
    RF = "RF"
    HGBC = "HGBC"
