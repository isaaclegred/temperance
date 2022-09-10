#!/usr/bin/env python
__usage__ = "setup.py command [--options]"
__description__ = "An interface for astrophysical data and inference of the nuclear EoS"

__author__ = "Isaac Legred (isaac.legred@caltech.edu)"

#-------------------------------------------------

from setuptools import (setup, find_packages)
import glob

setup(
    name = 'temperance',
    version = '0.0',
    author = __author__,
    description = __description__,
    license = 'MIT',
    scripts = glob.glob('bin/*'),
    packages = find_packages(),
    data_files = [],
    requires = ['numpy', 'pandas', 'universality',
                'scipy', 'bilby'],
)
