"""The setup script."""

from setuptools import setup

version = {}
with open("ccres_disdrometer_processing/__init__.py") as fp:
    exec(fp.read(), version)

setup(version=version["__version__"])
