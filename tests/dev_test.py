#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#
# (cfr.: http://num.pyro.ai/en/latest/examples/funnel.html)
#

# Installed without `-e` to smooth-out some PyCharm <-> WSL2 oddities
from nmc_numpyro import NMC

from jax import random
import jax.numpy as jnp

import numpyro
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

rng_key = random.PRNGKey(0)


# The used and abused Neal's Funnel
def model(dim=10):
    y = numpyro.sample("y", dist.Normal(0, 3))
    numpyro.sample("x", dist.Normal(jnp.zeros(dim - 1), jnp.exp(y / 2)))


nuts_kernel = NUTS(model)
nmc_kernel = NMC(model)

mcmc = MCMC(nmc_kernel, num_samples=20, num_warmup=20)

mcmc.run(rng_key)

mcmc.print_summary(exclude_deterministic=False)
