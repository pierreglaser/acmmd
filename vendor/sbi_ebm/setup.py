#!/usr/bin/env python
# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

description = (
    "Simulation Based Inference using Energy Based Models"
)

dist = setup(
    name="sbi_ebm",
    version="0.0.1dev0",
    description=description,
    license="BSD 3-Clause License",
    packages=["sbi_ebm.sbibm.tasks", "sbi_ebm.sbibm"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "License :: OSI Approved :: BSD License",
        "Operating System :: POSIX",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: Implementation :: CPython",
    ],
    python_requires=">=3.9",
)
