#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Imports
import jax.numpy as np
import numpyro.distributions as dist


def minka_normal(
    rng_key, x, grad, hess, normal_sigma_eigen_epsilon=1e-5, normal_sigma_correction=0.1
):
    ndim = np.ndim(x)
    if ndim == 0:
        inv_hess = np.divide(1, hess)
        dist_type = dist.Normal
    else:
        inv_hess = np.linalg.inv(hess)
        dist_type = dist.MultivariateNormal

    loc = np.subtract(x, np.dot(inv_hess, grad))
    sigma = np.negative(inv_hess)

    if not ndim == 0 and not np.all(np.linalg.eigvals(sigma) > 0):
        lamb, vec = np.linalg.eigh(sigma)
        sigma = vec @ np.diag(np.maximum(lamb, normal_sigma_eigen_epsilon)) @ vec.T

    dist_ = dist_type(loc, np.add(sigma, normal_sigma_correction))

    return dist_.sample(rng_key).reshape(x.shape), dist_


def minka_gamma(rng_key, x, grad, hess, gamma_concentration_correction=1.3):
    alpha = np.subtract(1, np.dot(np.power(x, 2), hess))
    beta = np.subtract(np.negative(np.dot(x, hess)), grad)

    dist_ = dist.continuous.Gamma(
        concentration=np.multiply(alpha, gamma_concentration_correction), rate=beta
    )

    return dist_.sample(rng_key).reshape(x.shape), dist_


def minka_dirichlet(rng_key, x, grad, hess, dirichlet_concentration_correction=1):
    _ = grad

    max_ndiag_hess = np.max(
        hess[np.logical_not(np.eye(hess.shape[0], dtype=bool))].reshape(
            hess.shape[0], -1
        ),
        axis=1,
    )

    concentration = np.subtract(
        1, np.multiply(np.power(x, 2), np.subtract(np.diag(hess), max_ndiag_hess))
    )

    dist_ = dist.Dirichlet(
        concentration=np.add(concentration, dirichlet_concentration_correction)
    )

    return dist_.sample(rng_key).reshape(x.shape), dist_


def minka_distribution_match(dist_):
    support = dist_.support
    if isinstance(
        support, (dist.constraints._Real, dist.constraints._IndependentConstraint)
    ):
        return minka_normal
    elif isinstance(support, dist.constraints._GreaterThan):
        return minka_gamma
    elif isinstance(support, dist.constraints._Simplex):
        return minka_dirichlet
    else:
        raise RuntimeError(
            "No valid proposal in Minka's 'distribution library' for support {}.".format(
                support
            )
        )
