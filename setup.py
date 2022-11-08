import os

import pkg_resources
from setuptools import setup, find_packages

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
    py_modules=["clip"],
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    include_package_data=True,
)