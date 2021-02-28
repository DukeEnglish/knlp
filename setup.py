#!/usr/bin/python
# -*- coding:UTF-8 -*-
# -----------------------------------------------------------------------#
# File Name: setup
# Author: Junyi Li
# Mail: 4ljy@163.com
# Created Time: 2021-01-30
# Description: 
# -----------------------------------------------------------------------#


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import setuptools

from setuptools import setup
from setuptools.command.develop import develop
from setuptools.command.install import install
from subprocess import call
from version import __version__
import distutils.text_file
from pathlib import Path
from typing import List

with open('README.md', 'r') as f:
    long_description = f.read()


def _parse_requirements(filename: str) -> List[str]:
    """Return requirements from requirements file."""
    # Ref: https://stackoverflow.com/a/42033122/
    return distutils.text_file.TextFile(filename=str(Path(__file__).with_name(filename))).readlines()


setuptools.setup(
    name='knlp',
    version=__version__,
    author='Junyi Li',
    author_email='ljyduke@gmail.com',
    description='KUAI SU(Quickly use) Python toolkit for Chinese Language Processing.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/DukeEnglish/knlp',
    package_data={'': ['*.md', '*.txt', '*.marshal', '*.marshal.3']},
    include_package_data=True,
    packages=setuptools.find_packages(exclude=('test*',)),
    install_requires=_parse_requirements('requirements.txt'),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'],
)
