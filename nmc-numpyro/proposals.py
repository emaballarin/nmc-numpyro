#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import jax.numpy as jnp
import numpyro.distributions as dists


def minka_normal(
    rng_key, x, grad, hess, normal_sigma_eigen_epsilon=1e-5, normal_sigma_correction=0.1
):
    ndim = jnp.ndim(x)
    if ndim == 0:
        inv_hess = jnp.divide(1, hess)
        dist_type = dists.Normal
    else:
        inv_hess = jnp.linalg.inv(hess)
        dist_type = dists.MultivariateNormal

    loc = jnp.subtract(x, jnp.dot(inv_hess, grad))
    sigma = jnp.negative(inv_hess)

    if not ndim == 0 and not jnp.all(jnp.linalg.eigvalsh(sigma) > 0):
        lamb, vec = jnp.linalg.eigh(sigma)
        sigma = vec @ jnp.diag(jnp.maximum(lamb, normal_sigma_eigen_epsilon)) @ vec.T

    dist = dist_type(loc, jnp.add(sigma, normal_sigma_correction))

    return dist.sample(rng_key).reshape(x.shape), dist


def minka_gamma(rng_key, x, grad, hess, gamma_concentration_correction=1.3):
    alpha = jnp.subtract(1, jnp.dot(jnp.power(x, 2), hess))
    beta = jnp.subtract(jnp.negative(jnp.dot(x, hess)), grad)

    dist = dists.continuous.Gamma(
        concentration=jnp.multiply(alpha, gamma_concentration_correction), rate=beta
    )

    return dist.sample(rng_key).reshape(x.shape), dist


def minka_dirichlet(rng_key, x, grad, hess, dirichlet_concentration_correction=1):
    _ = grad

    max_ndiag_hess = jnp.max(
        hess[jnp.logical_not(jnp.eye(hess.shape[0], dtype=bool))].reshape(
            hess.shape[0], -1
        ),
        axis=1,
    )

    concentration = jnp.subtract(
        1, jnp.multiply(jnp.power(x, 2), jnp.subtract(jnp.diag(hess), max_ndiag_hess))
    )

    dist = dists.Dirichlet(
        concentration=jnp.add(concentration, dirichlet_concentration_correction)
    )

    return dist.sample(rng_key).reshape(x.shape), dist


def minka_distribution_match(dist):
    support = dist.support
    if isinstance(
        support, (dists.constraints._Real, dists.constraints._IndependentConstraint)
    ):
        return minka_normal
    elif isinstance(support, dists.constraints._GreaterThan):
        return minka_gamma
    elif isinstance(support, dists.constraints._Simplex):
        return minka_dirichlet
    else:
        raise RuntimeError(
            "No valid proposal in Minka's 'distribution library' for support {}.".format(
                support
            )
        )
