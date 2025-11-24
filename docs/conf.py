import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

autodoc_mock_imports = [
    "torch",
    "torchvision",
    "pytorch_lightning",
    "numpy",
    "matplotlib",
    "PIL",
    "sklearn",
    "cv2",
    "albumentations",
    "torchmetrics",
    # add any other heavy deps used only in your code
]
