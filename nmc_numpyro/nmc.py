#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import Callable, Iterable

from collections import namedtuple
from copy import deepcopy

from jax import numpy as jnp
from jax import vmap, device_put, partial, grad, hessian
from jax import random as jrand

from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import init_to_uniform, log_density
from numpyro.handlers import seed, trace

# Make everything JIT-friendly
from numpyro.util import cond, identity

from .utils import initialize_model, ParamInfo, emplace_kv, rng_ckd_vec, kwargsify
from .proposals import minka_distribution_match

NMCState = namedtuple(
    "NMCState",
    ["i", "z", "mean_accept_prob", "rng_key"],
)


class NMC(MCMCKernel):
    def __init__(
        self, model, init_strategy=init_to_uniform, forward_mode_differentiation=False
    ):
        self._model = model
        self._init_strategy = init_strategy
        self._forward_mode_differentiation = forward_mode_differentiation

        # Set on first call to init
        self._distributions_at = None
        self._sample_single_site_fn = None
        self._sites_avail_keys = None
        self._sites_avail_len = None
        self._sample_fn = None

    @property
    def model(self):
        return self._model

    @property
    def distributions_at(self):
        return self._distributions_at

    @property
    def sites_avail_keys(self):
        return self._sites_avail_keys

    @property
    def sites_avail_len(self):
        return self._sites_avail_len

    @property
    def sample_field(self):
        return "z"

    def get_diagnostics_str(self, state):
        return "acc. prob={:.2f}".format(state.mean_accept_prob)

    def init(
        self,
        rng_key,
        num_warmup,
        init_params=None,
        permrange: Callable[[int], Iterable[int]] = range,
        model_args=(),
        model_kwargs=None,
    ):
        model_kwargs = kwargsify(model_kwargs)

        rng_key, rng_key_init_model = rng_ckd_vec(rng_key)

        if init_params is None:
            init_params, init_trace = initialize_model(
                rng_key_init_model,
                self._model,
                init_strategy=self._init_strategy,
                model_args=model_args,
                model_kwargs=model_kwargs,
                forward_mode_differentiation=self._forward_mode_differentiation,
            )
            self._distributions_at = deepcopy(
                {
                    k: init_trace[k]["fn"]
                    for k in init_trace.keys()
                    if init_trace[k]["type"] == "sample"
                }
            )
        else:
            if not isinstance(init_params, ParamInfo):
                init_params = ParamInfo(init_params)
                init_trace = trace(
                    seed(self.model, rng_key if jnp.ndim(rng_key) == 1 else rng_key[0])
                ).get_trace(*model_args, **model_kwargs)
                self._distributions_at = deepcopy(
                    {
                        k: init_trace[k]["fn"]
                        for k in init_trace.keys()
                        if init_trace[k]["type"] == "sample"
                    }
                )

        def nmc_init_fn(init_params, rng_key):
            minus_one_int = jnp.array(-1, dtype=jnp.result_type(int))
            zero_int = jnp.array(0, dtype=jnp.result_type(int))
            nmc_state = NMCState(minus_one_int, init_params, zero_int, rng_key)
            return device_put(nmc_state)

        def _sample_single_site_fn(
            nmc_state: NMCState, site_key, model_args, model_kwargs
        ) -> NMCState:
            def log_density_partialized(params):
                return partial(log_density, self.model, model_args, model_kwargs)(
                    params
                )[0]

            rng_key, rng_key_branched, rng_key_mh = rng_ckd_vec(nmc_state.rng_key, 2)

            # Useful self-diagnostics
            assert site_key in self.sites_avail_keys
            assert site_key in self.distributions_at.keys()

            # Acquire current (to-be previous) state
            given_state = nmc_state.z.z

            # Compute log-densities at current state
            given_ld = log_density_partialized(given_state)

            # Find a suitable proposal
            grad_to_match = grad(log_density_partialized)(given_state)[site_key]
            hess_to_match = hessian(log_density_partialized)(given_state)[site_key][
                site_key
            ]
            matched_proposal = minka_distribution_match(self.distributions_at[site_key])

            # Sample from such proposal
            site_val_proposal, site_fx_proposal = matched_proposal(
                rng_key_branched, given_state[site_key], grad_to_match, hess_to_match
            )

            # Compute the term "proposed distribution; given parameters" term
            # of Hastings criterion
            mixed_site_gppfx = site_fx_proposal.log_prob(given_state[site_key]).sum()

            # Sample a candidate state
            proposed_state = emplace_kv(given_state, site_key, site_val_proposal)

            # Compute log-densities at proposed state
            proposed_ld = log_density_partialized(proposed_state)
            proposed_site_ld = (
                self.distributions_at[site_key].log_prob(proposed_state[site_key]).sum()
            )

            # Compute acceptance probability
            exp_hastings_ratio = jnp.exp(
                proposed_ld + mixed_site_gppfx - given_ld - proposed_site_ld
            )

            acceptance_prob = cond(
                exp_hastings_ratio < 1.0, exp_hastings_ratio, identity, 1.0, identity
            )

            # Move, if the Metropolis-Hastings criterion agrees
            return cond(
                jrand.uniform(rng_key_mh) <= acceptance_prob,
                NMCState(
                    nmc_state.i,
                    ParamInfo(proposed_state),
                    nmc_state.mean_accept_prob + 1,  # Will be clearer later
                    rng_key,
                ),
                identity,
                NMCState(
                    nmc_state.i,
                    nmc_state.z,
                    nmc_state.mean_accept_prob,  # Will be clearer later
                    rng_key,
                ),
                identity,
            )

        self._sample_single_site_fn = _sample_single_site_fn

        def _sample_fn(
            nmc_state: NMCState,
            model_args,
            model_kwargs,
        ):

            nmc_state = NMCState(
                nmc_state.i,
                nmc_state.z,
                nmc_state.mean_accept_prob * (nmc_state.i + 1) * self.sites_avail_len,
                nmc_state.rng_key,
            )

            # For all sites, in given-permutation order
            for site_idx in permrange(self.sites_avail_len):

                # Sample from site
                nmc_state = self._sample_single_site_fn(
                    nmc_state, self.sites_avail_keys[site_idx], model_args, model_kwargs
                )

            # Second part of the trick above...
            return NMCState(
                nmc_state.i + 1,
                nmc_state.z,
                nmc_state.mean_accept_prob / ((nmc_state.i + 2) * self.sites_avail_len),
                nmc_state.rng_key,
            )

        self._sample_fn = _sample_fn

        # Vectorization of initialization/sampling
        # non-vectorized
        if rng_key.ndim == 1:
            init_state = nmc_init_fn(init_params, rng_key)
        # vectorized
        else:
            init_state = vmap(nmc_init_fn)(init_params, rng_key)
            sample_fn = vmap(self._sample_fn, in_axes=(0, None, None))
            self._sample_fn = sample_fn

        # Set model-parameter constants
        self._sites_avail_keys = tuple(init_state.z.z)
        self._sites_avail_len = len(self.sites_avail_keys)
        return init_state

    def sample(self, state, model_args=(), model_kwargs=None):
        model_kwargs = kwargsify(model_kwargs)
        return self._sample_fn(state, model_args, model_kwargs)
