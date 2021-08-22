#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
from copy import deepcopy

from jax import numpy as jnp
from jax import vmap, random, device_put, partial, grad, hessian

from numpyro.infer.mcmc import MCMCKernel
from numpyro.infer.util import init_to_uniform, log_density
from numpyro.handlers import seed, substitute, trace

from .utils import initialize_model, ParamInfo, emplace_kv
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
        self, rng_key, num_warmup, init_params=None, model_args=(), model_kwargs=None
    ):
        model_kwargs = {} if model_kwargs is None else model_kwargs

        # non-vectorized
        if rng_key.ndim == 1:
            rng_key, rng_key_init_model = random.split(rng_key)
        # vectorized
        else:
            rng_key, rng_key_init_model = jnp.swapaxes(
                vmap(random.split)(rng_key), 0, 1
            )

        #
        # Surrogate the call to _init_state(self, rng_key, model_args, model_kwargs, init_params)
        # Part 1: kernel initialization function
        #

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
                {k: init_trace[k]["fn"] for k in init_trace.keys()}
            )
        else:
            if not isinstance(init_params, ParamInfo):
                init_params = ParamInfo(init_params)
                init_trace = trace(
                    seed(self.model, rng_key if jnp.ndim(rng_key) == 1 else rng_key[0])
                ).get_trace(*model_args, **model_kwargs)
                self._distributions_at = deepcopy(
                    {k: init_trace[k]["fn"] for k in init_trace.keys()}
                )

        def nmc_init_fn(init_params, rng_key):
            minus_one_int = jnp.array(-1, dtype=jnp.result_type(int))
            zero_int = jnp.array(0, dtype=jnp.result_type(int))
            nmc_state = NMCState(minus_one_int, init_params, zero_int, rng_key)
            return device_put(nmc_state)

        #
        # Surrogate the call to _init_state(self, rng_key, model_args, model_kwargs, init_params)
        # Part 2: kernel sampling function
        #

        def _sample_single_site_fn(
            nmc_state: NMCState, site_key, model_args=(), model_kwargs=None
        ):
            # TODO: cleanup comments!

            # Shorthand partialization
            def log_density_partialized(params):
                return partial(log_density, self.model, model_args, model_kwargs)(
                    params
                )[0]

            # TODO: Do the split in 3, in the proper way!

            # non-vectorized
            if nmc_state.rng_key.ndim == 1:
                rng_key, rng_key_branched = random.split(nmc_state.rng_key)
                rng_key, rng_key_mh = random.split(rng_key)
            # vectorized
            else:
                rng_key, rng_key_branched = jnp.swapaxes(
                    vmap(random.split)(nmc_state.rng_key), 0, 1
                )
                rng_key, rng_key_mh = jnp.swapaxes(vmap(random.split)(rng_key), 0, 1)

            # Diag
            assert site_key in self.sites_avail_keys
            assert site_key in self.distributions_at.keys()

            # 3) #######
            #
            # Compute log density for given state
            given_state = nmc_state.z.z
            given_ld = log_density_partialized(given_state)
            given_site_lp = self.distributions_at[site_key].log_prob()  # FIXME: HM!
            #
            # Find suitable proposal
            grad_to_match = grad(log_density_partialized)(given_state)[site_key]
            hess_to_match = hessian(log_density_partialized)(given_state)[site_key][
                site_key
            ]
            matched_proposal = minka_distribution_match(self.distributions_at[site_key])
            #
            # Sample from proposal
            site_val_proposal, _ = matched_proposal(
                rng_key_branched, given_state[site_key], grad_to_match, hess_to_match
            )
            #
            # Create a candidate state
            proposed_state = emplace_kv(given_state, site_key, site_val_proposal)
            #
            # Compute log density for proposed state
            proposed_ld = log_density_partialized(proposed_state)
            #
            #
            # ##########
            return None

        self._sample_single_site_fn = _sample_single_site_fn

        def _sample_fn(nmc_state: NMCState, model_args=(), model_kwargs=None):
            model_kwargs = {} if model_kwargs is None else model_kwargs

            # Eventually vectorize
            rng_key, rng_key_sample, rng_key_accept = random.split(nmc_state.rng_key, 3)

            # Prepare next state
            new_i = nmc_state.i + 1
            new_rng_key = rng_key

            # TESTS
            sites_avail = nmc_state.z.z
            site_sampl = self.sites_avail_keys[new_i % self._sites_avail_len]

            ld = log_density(self.model, model_args, model_kwargs, sites_avail)[0]
            print(ld)

            sites_prop = emplace_kv(
                sites_avail, site_sampl, jnp.zeros_like(sites_avail.get(site_sampl))
            )

            ld = log_density(self.model, model_args, model_kwargs, sites_prop)[0]
            print(ld)

            # TODO: remove ->
            new_z = nmc_state.z
            new_mean_acc_prob = nmc_state.mean_accept_prob
            # TODO: remove <-

            return NMCState(new_i, new_z, new_mean_acc_prob, new_rng_key)

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
        return self._sample_fn(state, model_args, model_kwargs)
