from collections import namedtuple

from jax.random import PRNGKey, split
import jax.numpy as np
from jax import partial, grad, hessian

from numpyro.infer.util import log_density
import numpyro.distributions as dist
from numpyro import sample
from numpyro.handlers import trace, substitute, replay

from .proposals import minka_distribution_match
from .utils import filter_unobserved_sites


# Tracks state of the NMC algorithm
NMCState = namedtuple(
    "NMCState", ["i", "params", "log_likelihood", "accept_prob", "rng_key"]
)


class NMC:
    # Class containing all of the NMC functionality
    """
    The class containing all NMC functionality. This includes the main sampling loop,
    determination of support, individual proposal distributions and functions for running the algorithm
    """
    # Init the classe with a model and the data

    def __init__(self, model, *model_args, rng_key=PRNGKey(0), **model_kwargs):
        self.model = model
        self.model_args = model_args
        self.rng_key = rng_key
        self.model_kwargs = model_kwargs

        tr = trace(model).get_trace(model_args)
        log_likelihood = log_density(
            self.model, self.model_args, self.model_kwargs, filter_unobserved_sites(tr)
        )[0]

        self.nmc_state = NMCState(
            i=0,
            params=filter_unobserved_sites(tr),
            log_likelihood=log_likelihood,
            accept_prob=0.0,
            rng_key=rng_key,
        )

        self.props = {}
        self.acc_trace = {}
        self.init_trace()

    # Initialize the accepted trace proposal objects

    def init_trace(self):
        for name in self.nmc_state.params:
            dim = len(self.nmc_state.params[name])
            for i in range(dim):
                self.props[name + str(i)] = []
                self.acc_trace[name + str(i)] = []

    # The core sampler functions. Running a a single site inferece MH algorithm

    def sample(self):
        rng_key, rng_key_sample, rng_key_accept = split(self.nmc_state.rng_key, 3)
        params = self.nmc_state.params

        for site in params.keys():
            # Collect accepted trace
            for i in range(len(params[site])):
                self.acc_trace[site + str(i)].append(params[site][i])

            tr_current = trace(substitute(self.model, params)).get_trace(
                *self.model_args, **self.model_kwargs
            )
            ll_current = self.nmc_state.log_likelihood

            val_current = tr_current[site]["value"]
            dist_curr = tr_current[site]["fn"]

            def log_den_fun(var):
                return partial(
                    log_density, self.model, self.model_args, self.model_kwargs
                )(var)[0]

            val_proposal, dist_proposal = self.proposal(
                site,
                log_den_fun,
                filter_unobserved_sites(tr_current),
                dist_curr,
                rng_key_sample,
            )

            tr_proposal = self.retrace(
                site,
                tr_current,
                dist_proposal,
                val_proposal,
                self.model_args,
                self.model_kwargs,
            )
            ll_proposal = log_density(
                self.model,
                self.model_args,
                self.model_kwargs,
                filter_unobserved_sites(tr_proposal),
            )[0]

            ll_proposal_val = dist_proposal.log_prob(val_current).sum()
            ll_current_val = dist_curr.log_prob(val_proposal).sum()

            hastings_ratio = (ll_proposal + ll_proposal_val) - (
                ll_current + ll_current_val
            )

            accept_prob = np.minimum(1, np.exp(hastings_ratio))
            u = sample("u", dist.Uniform(0, 1), rng_key=rng_key_accept)

            if u <= accept_prob:
                params, ll_current = filter_unobserved_sites(tr_proposal), ll_proposal
            else:
                params, ll_current = filter_unobserved_sites(tr_current), ll_current

        iter = self.nmc_state.i + 1
        mean_accept_prob = (
            self.nmc_state.accept_prob
            + (accept_prob - self.nmc_state.accept_prob) / iter
        )

        return NMCState(iter, params, ll_current, mean_accept_prob, rng_key)

    # Computes the gradient and hessian to use in the specific proposal function

    def proposal(self, name, log_den_fun, params, dist_curr, rng_key):
        grad_fn = grad(log_den_fun)
        hess_fn = hessian(log_den_fun)

        grad_ = grad_fn(params)[name]
        hess = hess_fn(params)[name][name]

        proposal_ = minka_distribution_match(dist_curr)
        value, dist_ = proposal_(rng_key, params[name], grad_, hess)

        # Collect proposals
        dim = len(value)
        for i in range(dim):
            var_name = name + str(i)
            self.props[var_name].append(value[i])

        return value, dist_

    # Reruns a trace with the new proposed value and distribution
    def retrace(self, name, tr, dist_proposal, val_proposal, model_args, model_kwargs):
        fn_current = tr[name]["fn"]
        val_current = tr[name]["value"]

        tr[name]["fn"] = dist_proposal
        tr[name]["value"] = val_proposal

        second_trace = trace(replay(self.model, tr)).get_trace(
            *model_args, **model_kwargs
        )

        tr[name]["fn"] = fn_current
        tr[name]["value"] = val_current

        return second_trace

    # Run the inferece with number of iterations

    def run(self, iterations=1000):
        while self.nmc_state.i < iterations:
            self.nmc_state = self.sample()

        # Collect last trace
        for site in self.nmc_state.params.keys():
            for i in range(len(self.nmc_state.params[site])):
                self.acc_trace[site + str(i)].append(self.nmc_state.params[site][i])

        return self.nmc_state
