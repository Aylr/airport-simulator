# !/usr/bin/env python

from distutils.core import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="simulator",
    packages=["simulator"],
    version="0.0.1",
    description="airline simulator",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="Author",
    url="https://github.com",
    keywords=[""],
    install_requires=[],
    scripts=[],
    include_package_data=True,
)
