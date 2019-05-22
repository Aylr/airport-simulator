# !/usr/bin/env python

from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="simulator",
    packages=["simulator"],
    version="1.2.0",
    description="Airline simulator for saving the earth by decreasing wasted jet fuel with suboptimal stand loading.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Taylor Miller",
    url="https://github.com/Aylr",
    keywords=[""],
    install_requires=["mesa", "matplotlib"],
    scripts=[],
    include_package_data=True,
)
