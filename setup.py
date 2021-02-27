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

with open('README.md', 'r') as f:
    long_description = f.read()


class Installation(install):
    def run(self):
        call(['pip install -r requirements.txt --no-clean'], shell=True)
        install.run(self)


setuptools.setup(
    name='knlp',
    version='0.1.0',
    author='Junyi Li',
    author_email='ljyduke@gmail.com',
    description='KUAI SU(Quickly use) Python toolkit for Chinese Language Processing.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/DukeEnglish/knlp',
    package_data={'': ['*.md', '*.txt', '*.marshal', '*.marshal.3']},
    include_package_data=True,
    packages=setuptools.find_packages(exclude=('test*', )),
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent'],
    cmdclass={'install': Installation})
