import importlib.util
import pathlib


ROOT = pathlib.Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "predictor.py"


def test_predictor_module_exists():
    assert MODULE_PATH.exists(), "predictor.py should exist in the model folder"


def test_predictor_module_can_be_loaded():
    spec = importlib.util.spec_from_file_location("predictor", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert hasattr(module, "load_model")
    assert hasattr(module, "load_class_names")
    assert hasattr(module, "preprocess_image")
    assert hasattr(module, "predict_image")
