#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
from setuptools import find_packages, setup

package = 'Kinho'

with open(os.path.join(package, '__init__.py'), 'rb') as f:
    init_py = f.read().decode('utf-8')

version = re.search(
    '^__version__ = [\'\"]([^\'\"]+)[\'\"]', init_py, re.MULTILINE
).group(1)
author = re.search(
    '^__author__ = [\'\"]([^\'\"]+)[\'\"]', init_py, re.MULTILINE
).group(1)
email = re.search(
    '^__email__ = [\'\"]([^\'\"]+)[\'\"]', init_py, re.MULTILINE
).group(1)

packages = find_packages()
packages.remove('tests')

setup(
    name='Kinho',
    packages=packages,
    version=version,
    description='A library to classify images with deep learning.',
    long_description='The library features the "Neural" model, which is a Convolutional Neural ' \
        + 'Network (CNN) for image classification. It supports both CPU and GPU, providing excellent performance ' \
        + 'for large networks. You can also export and import Deep models and continue training on other machines. ' \
        + 'The exported file is generated in the ".brain" format, which is a proprietary data type of this project. '\
        + 'For more information, please visit our repository.',
    author=author,
    author_email=email,
    url='https://github.com/kinhosz/Neural',
    install_requires=[],
    license='MIT',
    keywords=['dev', 'web'],
    classifiers=[
       'Development Status :: 5 - Production/Stable',
       'Environment :: GPU',
       'Environment :: GPU :: NVIDIA CUDA',
       'Environment :: GPU :: NVIDIA CUDA :: 12 :: 12.0',
       'License :: OSI Approved :: MIT License',
       'Natural Language :: Portuguese (Brazilian)',
       'Natural Language :: English',
       'Programming Language :: Python',
       'Programming Language :: Python :: 3',
       'Topic :: Scientific/Engineering :: Artificial Intelligence',
       'Topic :: Scientific/Engineering :: Image Processing',
       'Topic :: Scientific/Engineering :: Image Recognition',
    ],
)