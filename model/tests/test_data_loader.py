import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "train_pipeline.py"


def test_module_exists():
    assert MODULE_PATH.exists(), "train_pipeline.py should exist in the model folder"


def test_module_can_be_loaded():
    spec = importlib.util.spec_from_file_location("train_pipeline", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "build_model")
    assert hasattr(module, "prepare_dataset")
