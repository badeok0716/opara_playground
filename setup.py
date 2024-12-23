from setuptools import setup, find_packages
import os
import sys

_here = os.path.abspath(os.path.dirname(__file__))

if sys.version_info[0] < 3:
    with open(os.path.join(_here, "README.md")) as f:
        long_description = f.read()
else:
    with open(os.path.join(_here, "README.md"), encoding="utf-8") as f:
        long_description = f.read()

desc = 'GitHub repository for Opara'

setup(
    name="Opara",
    version="0.1.0",
    description=desc,
    long_description=long_description,
    packages=find_packages(),
)