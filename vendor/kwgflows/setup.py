#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

description = (
    "Wasserstein Gradient Flows of kernel-based divergences"
)

dist = setup(
    name="kwgflows",
    version="0.0.1dev0",
    description=description,
    license="BSD 3-Clause License",
    packages=[
        "kwgflows",
        "kwgflows.rkhs",
        "kwgflows.divergences",
        "kwgflows.gradient_flow",
    ],
    # install_requires=["jax<0.4.14"],
    install_requires=["jax"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Mathematics",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires=">=3.9",
)
