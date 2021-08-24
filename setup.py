#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ==============================================================================
#
# Copyright 2019-* Yana Hasson <yana.hasson.inria@gmail.com>
# Copyright 2019-* Linxi (Jim) Fan <jimfanspire@gmail.com>
#
# ==============================================================================
#
# Copyright 2020-* Emanuele Ballarin <emanuele@ballarin.cc>
# All Rights Reserved. Unless otherwise explicitly stated.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================
#
# SPDX-License-Identifier: Apache-2.0

import os
import warnings
from setuptools import find_packages, setup


def read(fname):
    with open(os.path.join(os.path.dirname(__file__), fname)) as f:
        return f.read().strip()


def check_dependencies():
    missing_dependencies = []
    for package_name in DEPENDENCY_PACKAGE_NAMES:
        try:
            __import__(package_name)
        except ImportError:
            missing_dependencies.append(package_name)

    if missing_dependencies:
        warnings.warn("Missing dependencies: {}.".format(missing_dependencies))


DEPENDENCY_PACKAGE_NAMES = ["jaxlib>=0.1.68", "jax>=0.2.16", "numpyro>=0.6.0"]

check_dependencies()


setup(
    name="nmc_numpyro",
    version="0.0.1",
    author="<at>Johanpdrsn & Emanuele Ballarin",
    author_email="emanuele@ballarin.cc",
    url="https://github.com/emaballarin/nmc-numpyro",
    description="Newtonian Monte-Carlo sampling algorithm, implemented in JAX/NumPyro (after Arora et al., 2020)",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    keywords=[
        "Deep Learning",
        "Machine Learning",
        "Statistics",
        "Bayesian Inference",
        "Sampling",
        "Markov Chain Monte Carlo",
    ],
    license="Custom",
    packages=[
        package for package in find_packages() if package.startswith("nmc_numpyro")
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Environment :: Console",
        "Programming Language :: Python :: 3",
    ],
    python_requires=">=3.8",
    include_package_data=True,
    zip_safe=False,
)
