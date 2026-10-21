from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class AppConfig:
    data_dir: Path = Path("data")
    model_dir: Path = Path("models")
    model_path: Path = Path("models/cavity_cnn.pt")
    labels_csv: Path = Path("labels.csv")
    image_size: int = 224
    default_threshold: float = 0.5

