# Copyright 2022 Artificial Intelligence and Machine Learning Lab @ TU Darmstadt. All rights reserved.
#
# Licensed under the GNU General Public License Version 3 (the "License");
# Everyone is permitted to copy and distribute verbatim copies
# of this license document, but changing it is not allowed.

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from distutils.core import Command

from setuptools import find_packages, setup


# IMPORTANT:
# 1. all dependencies should be listed here with their version requirements if any
# 2. once modified, run: `make deps_table_update` to update src/diffusers/dependency_versions_table.py
_deps = [
    "Pillow<10.0",  # keep the PIL.Image.Resampling deprecation away
    "accelerate>=0.11.0",
    "black==22.8",
    "datasets",
    "filelock",
    "flake8>=3.8.3",
    "flax>=0.4.1",
    "hf-doc-builder>=0.3.0",
    "huggingface-hub>=0.10.0",
    "importlib_metadata",
    "isort>=5.5.4",
    "jax>=0.2.8,!=0.3.2",
    "jaxlib>=0.1.65",
    "modelcards>=0.1.4",
    "numpy",
    "parameterized",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "scipy",
    "regex!=2019.12.17",
    "requests",
    "tensorboard",
    "torch>=1.4",
    "torchvision",
    "transformers>=4.21.0",
]

# this is a lookup table with items like:
#
# tokenizers: "huggingface-hub==0.8.0"
# packaging: "packaging"
#
# some of the values are versioned whereas others aren't.
deps = {b: a for a, b in (re.findall(r"^(([^!=<>~]+)(?:[!=<>~].*)?$)", x)[0] for x in _deps)}

# since we save this data in src/diffusers/dependency_versions_table.py it can be easily accessed from
# anywhere. If you need to quickly access the data from this table in a shell, you can do so easily with:
#
# python -c 'import sys; from diffusers.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets
#
# Just pass the desired package names to that script as it's shown with 2 packages above.
#
# If diffusers is not yet installed and the work is done from the cloned repo remember to add `PYTHONPATH=src` to the script above
#
# You can then feed this for example to `pip`:
#
# pip install -U $(python -c 'import sys; from diffusers.dependency_versions_table import deps; \
# print(" ".join([ deps[x] for x in sys.argv[1:]]))' tokenizers datasets)
#


def deps_list(*pkgs):
    return [deps[pkg] for pkg in pkgs]


extras = {}


extras = {}
extras["quality"] = deps_list("black", "isort", "flake8", "hf-doc-builder")
extras["docs"] = deps_list("hf-doc-builder")
extras["training"] = deps_list("accelerate", "datasets", "tensorboard", "modelcards")
extras["test"] = deps_list(
    "datasets",
    "parameterized",
    "pytest",
    "pytest-timeout",
    "pytest-xdist",
    "scipy",
    "torchvision",
    "transformers"
)
extras["torch"] = deps_list("torch", "accelerate")

if os.name == "nt":  # windows
    extras["flax"] = []  # jax is not supported on windows
else:
    extras["flax"] = deps_list("jax", "jaxlib", "flax")

extras["dev"] = (
    extras["quality"] + extras["test"] + extras["training"] + extras["docs"] + extras["torch"] + extras["flax"]
)

install_requires = [
    deps["importlib_metadata"],
    deps["filelock"],
    deps["huggingface-hub"],
    deps["numpy"],
    deps["regex"],
    deps["requests"],
    deps["Pillow"],
]

setup(
    name="sld",
    version="0.0.1",  # expected format is one of x.y.z.dev0, or x.y.z.rc1 or x.y.z (no to dashes, yes to dots)
    description="Safe Latent Diffusion",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="deep learning",
    license="GNU",
    author="Manuel Brack",
    author_email="brac@cs.tu-darmstadt.de",
    url="https://github.com/ml-research/safe-latent-diffusion",
    package_dir={"": "src"},
    packages=find_packages("src"),
    include_package_data=True,
    python_requires=">=3.7.0",
    install_requires=install_requires,
    extras_require=extras,
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: GNU General Public License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
