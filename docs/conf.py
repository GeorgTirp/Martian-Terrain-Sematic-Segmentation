import os
import sys

# Point to repo root and src
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
]
