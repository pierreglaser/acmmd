#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re

try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup


dist = setup(
    name='jax_samplers',
    version='0.1.0dev0',
    license='BSD 3-Clause License',
    packages=[
        "jax_samplers",
        "jax_samplers.inference_algorithms",
        "jax_samplers.inference_algorithms.mcmc",
        "jax_samplers.inference_algorithms.importance_sampling",
        "jax_samplers.kernels",
    ],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: BSD License',
        'Operating System :: POSIX',
        'Operating System :: Microsoft :: Windows',
        'Operating System :: MacOS :: MacOS X',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: Implementation :: CPython',
        'Programming Language :: Python :: Implementation :: PyPy',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Scientific/Engineering',
        'Topic :: System :: Distributed Computing',
    ],
    test_suite='tests',
    python_requires='>=3.9',
    install_requires=['jax', 'flax', 'numpyro','blackjax<1', 'typing_extensions'],
)
