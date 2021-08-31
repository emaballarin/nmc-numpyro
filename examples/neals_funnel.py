#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# An implementation of Neal's Funnel with NUTS and NMC kernels.
# Full API-compatibility of NMC with NumPyro's MCMC ABCs.
#
# (cfr.: http://num.pyro.ai/en/latest/examples/funnel.html)
#

import os

from jax import random
import jax.numpy as jnp
from jax.config import config

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

from nmc_numpyro import NMC


# Usual workarounds to force-enable JIT and avoid GPU OOMs
config.update("jax_disable_jit", False)
os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".84"

# Set initial random seed
rng_key = random.PRNGKey(0)


# Actual Neal's Funnel model (10-D): no modification from w.r.t. original code
def model(dim=10):
    y = numpyro.sample("y", dist.Normal(0, 3))
    numpyro.sample("x", dist.Normal(jnp.zeros(dim - 1), jnp.exp(y / 2)))


# Kernel instantiation: exact same API
nuts_kernel = NUTS(model)
nmc_kernel = NMC(model)


# "Run Markov Chain, run!"
nuts_mcmc_runner = MCMC(nuts_kernel, num_samples=200, num_warmup=200)
nmc_mcmc_runner = MCMC(nmc_kernel, num_samples=200, num_warmup=200)

# Compare
print("<><><><><><><><><><><><><><><><><><><><><><><><>")
nuts_mcmc_runner.run(rng_key)
nuts_mcmc_runner.print_summary(exclude_deterministic=True)
print("<><><><><><><><><><><><><><><><><><><><><><><><>")
nmc_mcmc_runner.run(rng_key)
nmc_mcmc_runner.print_summary(exclude_deterministic=True)
print("<><><><><><><><><><><><><><><><><><><><><><><><>")
