#!/usr/bin/env python
# -*- coding: utf-8 -*-

from setuptools import find_packages, setup

setup(
    name='Kinho',
    packages=find_packages(),
    version='2.1.0',
    description='A library to classify images with deep learning.',
    long_description='The library features the "Neural" model, which is a Convolutional Neural ' \
        + 'Network (CNN) for image classification. It supports both CPU and GPU, providing excellent performance ' \
        + 'for large networks. You can also export and import Deep models and continue training on other machines. ' \
        + 'The exported file is generated in the ".brain" format, which is a proprietary data type of this project. '\
        + 'For more information, please visit our repository.',
    author='kinhosz',
    author_email='scruz.josecarlos@gmail.com',
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
    ],
)