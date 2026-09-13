import torch.nn as nn


class LightningModule(nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__()

    def save_hyperparameters(self, *args, **kwargs):
        return None

    def log(self, *args, **kwargs):
        return None


class Callback:
    pass


class Trainer:
    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.progress_bar_refresh_rate = 0

    def fit(self, *args, **kwargs):
        raise NotImplementedError(
            "Local pytorch_lightning shim only supports checkpoint loading/inference."
        )
