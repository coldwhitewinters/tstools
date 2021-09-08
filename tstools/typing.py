from typing_extensions import Protocol


class ScikitModel(Protocol):
    def fit(self, X, y, sample_weight=None): ...
    def predict(self, X): ...
    def score(self, X, y, sample_weight=None): ...
    def set_params(self, **params): ...
    def get_params(self, deep=True): ...


class ScikitScaler(Protocol):
    def fit(self, X, y=None, sample_weight=None): ...
    def transform(self, X, y=None): ...
    def set_params(self, **params): ...
    def get_params(self, deep=True): ...
