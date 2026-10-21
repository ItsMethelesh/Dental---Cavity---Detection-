"""Dental cavity detection package."""

__all__ = ["load_model_bundle", "predict_scan", "train_model"]


def load_model_bundle(*args, **kwargs):
    from .inference import load_model_bundle as _load_model_bundle

    return _load_model_bundle(*args, **kwargs)


def predict_scan(*args, **kwargs):
    from .inference import predict_scan as _predict_scan

    return _predict_scan(*args, **kwargs)


def train_model(*args, **kwargs):
    from .train import train_model as _train_model

    return _train_model(*args, **kwargs)
