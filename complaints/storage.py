"""Saving a trained classifier, so it doesn't have to learn again every time.

Training on a big file can take minutes. Saving the result is like saving
a game: next time you carry on from where you were.

Python's pickle is used to save it. Warning: opening a pickle file can run
code hidden inside it, so only ever load model files you made yourself,
never ones downloaded from someone else.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path

from .classifier import Classifier
from .exceptions import DataError

FORMAT = "complaints-model"
VERSION = 1


def save_model(classifier: Classifier, path: str | Path) -> Path:
    if not classifier.labels:
        raise ValueError("Train the classifier before saving it.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    package = {"format": FORMAT, "version": VERSION, "saved_at": datetime.now().isoformat(),
               "classifier": classifier}
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("wb") as file:
        pickle.dump(package, file)
    temp.replace(path)     # swap in only when it's fully written
    return path


def load_model(path: str | Path) -> Classifier:
    path = Path(path)
    try:
        with path.open("rb") as file:
            package = pickle.load(file)
    except FileNotFoundError:
        raise DataError(f"No model at {path}. Train one first.") from None
    except (pickle.UnpicklingError, EOFError, AttributeError, ImportError) as error:
        raise DataError(f"{path} isn't a model this program can read: {error}") from None
    if not isinstance(package, dict) or package.get("format") != FORMAT:
        raise DataError(f"{path} isn't a complaints model file.")
    if package.get("version") != VERSION:
        raise DataError(f"{path} was saved by a different version. Train it again.")
    classifier = package.get("classifier")
    if not isinstance(classifier, Classifier):
        raise DataError(f"{path} doesn't hold a classifier.")
    return classifier
