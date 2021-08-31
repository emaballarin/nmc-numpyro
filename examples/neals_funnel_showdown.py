#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import os
import matplotlib.pyplot as plt
from jax import random
import jax.numpy as jnp
from jax.config import config
import numpyro
import numpyro.distributions as dist
from numpyro.handlers import reparam
from numpyro.infer import MCMC, NUTS, Predictive
from nmc_numpyro import NMC
from numpyro.infer.reparam import LocScaleReparam


# The model itself
def model(dim=10):
    y = numpyro.sample("y", dist.Normal(0, 3))
    numpyro.sample("x", dist.Normal(jnp.zeros(dim - 1), jnp.exp(y / 2)))


# The automatically-reparameterized model (after Gorinova et al., 2020)
reparam_model = reparam(model, config={"x": LocScaleReparam(0)})


# Wrapper functions
def run_inference(model, kernel_fx, rng_key):
    kernel = kernel_fx(model)
    mcmc = MCMC(
        kernel,
        # Edit directly here!
        num_warmup=1000,
        num_samples=25000,
        num_chains=1,
        progress_bar=True,
    )
    mcmc.run(rng_key)
    return mcmc.get_samples()


def run_nuts_vanilla(rng_key):
    return run_inference(model, NUTS, rng_key)


def run_nuts_reparam(rng_key):
    return run_inference(reparam_model, NUTS, rng_key)


def run_nmc_vanilla(rng_key):
    return run_inference(model, NMC, rng_key)


def run_nmc_reparam(rng_key):
    return run_inference(reparam_model, NMC, rng_key)


# Main function


def main():
    initial_rng_key = random.PRNGKey(0)
    initial_rng_key_p = random.PRNGKey(1)

    # NUTS,vanilla
    nuv = run_nuts_vanilla(initial_rng_key)

    # NUTS, reparameterized
    nur = run_nuts_reparam(initial_rng_key)
    nurp = Predictive(reparam_model, nur, return_sites=["x", "y"])(initial_rng_key_p)

    # NMC, vanilla
    nmv = run_nmc_vanilla(initial_rng_key).z

    # NMC, reparameterized
    nmr = run_nmc_reparam(initial_rng_key).z
    nmrp = Predictive(reparam_model, nmr, return_sites=["x", "y"])(initial_rng_key_p)

    #
    # PLOTTING
    #

    # NUTS vs Reparameterized NUTS
    fig1, (ax1, ax2) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 8), constrained_layout=True
    )

    ax1.plot(nuv["x"][:, 0], nuv["y"], "go", alpha=0.3)
    ax1.set(
        xlim=(-20, 20),
        ylim=(-9, 9),
        xlabel="x[0]",
        ylabel="y",
        title="Funnel samples: NUTS, centered parameterization",
    )

    ax2.plot(nurp["x"][:, 0], nurp["y"], "go", alpha=0.3)
    ax2.set(
        xlim=(-20, 20),
        ylim=(-9, 9),
        xlabel="x[0]",
        ylabel="y",
        title="Funnel samples: NUTS, non-centered parameterization",
    )

    plt.savefig("imgs/funnel_NUTS.png")

    # NMC vs Reparameterized NMC
    fig2, (ax3, ax4) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 8), constrained_layout=True
    )

    ax3.plot(nmv["x"][:, 0], nmv["y"], "go", alpha=0.3)
    ax3.set(
        xlim=(-20, 20),
        ylim=(-9, 9),
        xlabel="x[0]",
        ylabel="y",
        title="Funnel samples: NMC, centered parameterization",
    )

    ax4.plot(nmrp["x"][:, 0], nmrp["y"], "go", alpha=0.3)
    ax4.set(
        xlim=(-20, 20),
        ylim=(-9, 9),
        xlabel="x[0]",
        ylabel="y",
        title="Funnel samples: NMC, non-centered parameterization",
    )

    plt.savefig("imgs/funnel_NMC.png")

    # NUTS vs NMC
    fig3, (ax5, ax6) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 8), constrained_layout=True
    )

    ax5.plot(nuv["x"][:, 0], nuv["y"], "go", alpha=0.3)
    ax5.set(
        xlim=(-20, 20),
        ylim=(-9, 9),
        xlabel="x[0]",
        ylabel="y",
        title="Funnel samples: NUTS, centered parameterization",
    )

    ax6.plot(nmv["x"][:, 0], nmv["y"], "go", alpha=0.3)
    ax6.set(
        xlim=(-20, 20),
        ylim=(-9, 9),
        xlabel="x[0]",
        ylabel="y",
        title="Funnel samples: NMC, centered parameterization",
    )

    plt.savefig("imgs/funnel_centered.png")

    # Reparameterized (NUTS vs NMC)
    fig4, (ax7, ax8) = plt.subplots(
        2, 1, sharex=True, figsize=(8, 8), constrained_layout=True
    )

    ax7.plot(nurp["x"][:, 0], nurp["y"], "go", alpha=0.3)
    ax7.set(
        xlim=(-20, 20),
        ylim=(-9, 9),
        xlabel="x[0]",
        ylabel="y",
        title="Funnel samples: NUTS, non-centered parameterization",
    )

    ax8.plot(nmrp["x"][:, 0], nmrp["y"], "go", alpha=0.3)
    ax8.set(
        xlim=(-20, 20),
        ylim=(-9, 9),
        xlabel="x[0]",
        ylabel="y",
        title="Funnel samples: NMC, non-centered parameterization",
    )

    plt.savefig("imgs/funnel_noncentered.png")


if __name__ == "__main__":

    # Usual workarounds to force-enable JIT and avoid GPU OOMs
    config.update("jax_disable_jit", False)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = ".84"

    # Edit here!
    numpyro.set_platform("cpu")
    numpyro.set_host_device_count(1)

    main()
