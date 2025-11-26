import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC = os.path.join(ROOT, "src")
sys.path.insert(0, SRC)

project = "Martian Terrain Semantic Segmentation"
author = "Georg Tirp"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
]

# If you want type hints rendered nicely
autodoc_typehints = "description"

# -------------------------------------------------------------------
# Mock heavy / unavailable dependencies *only on CI*, not locally
# -------------------------------------------------------------------
if os.environ.get("GITHUB_ACTIONS") == "true":
    autodoc_mock_imports = [
        "torch",
        "torchvision",
        "pytorch_lightning",
        "segmentation_models_pytorch",
        "albumentations",
        "cv2",
        "sklearn",
        "skimage",
        "PIL",
        "matplotlib",
        # add anything else that fails to import in the docs-only env
    ]
else:
    # Locally, don’t mock anything so you can see real errors.
    autodoc_mock_imports = []