#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from collections import namedtuple
from functools import partial
import warnings

from jax import device_get
import jax.numpy as jnp

import numpyro
from numpyro.handlers import seed, substitute, trace
from numpyro.infer.initialization import init_to_uniform, init_to_value
from numpyro.util import not_jax_tracer

from numpyro.infer.util import (
    _get_model_transforms,
    _guess_max_plate_nesting,
    _validate_model,
    transform_fn,
    _init_to_unconstrained_value,
    find_valid_initial_params,
)

__all__ = ["initialize_model", "emplace_kv"]

ModelInfo = namedtuple("ModelInfo", ["param_info", "model_trace"])
ParamInfo = namedtuple("ParamInfo", ["z"])


def emplace_kv(dictionary: dict, k, v) -> dict:
    """
    Returns input dict with added k:v pair, overwriting if k already exists
    """
    return {**dictionary, k: v}


def initialize_model(
    rng_key,
    model,
    *,
    init_strategy=init_to_uniform,
    model_args=(),
    model_kwargs=None,
    forward_mode_differentiation=False,
):

    model_kwargs = {} if model_kwargs is None else model_kwargs
    substituted_model = substitute(
        seed(model, rng_key if jnp.ndim(rng_key) == 1 else rng_key[0]),
        substitute_fn=init_strategy,
    )
    (
        inv_transforms,
        replay_model,
        has_enumerate_support,
        model_trace,
    ) = _get_model_transforms(substituted_model, model_args, model_kwargs)
    # substitute param sites from model_trace to model so
    # we don't need to generate again parameters of `numpyro.module`
    model = substitute(
        model,
        data={
            k: site["value"]
            for k, site in model_trace.items()
            if site["type"] in ["param"]
        },
    )
    constrained_values = {
        k: v["value"]
        for k, v in model_trace.items()
        if v["type"] == "sample" and not v["is_observed"] and not v["fn"].is_discrete
    }

    if has_enumerate_support:
        from numpyro.contrib.funsor import config_enumerate, enum

        if not isinstance(model, enum):
            max_plate_nesting = _guess_max_plate_nesting(model_trace)
            _validate_model(model_trace)
            model = enum(config_enumerate(model), -max_plate_nesting - 1)

    init_strategy = (
        init_strategy if isinstance(init_strategy, partial) else init_strategy()
    )
    if (init_strategy.func is init_to_value) and not replay_model:
        init_values = init_strategy.keywords.get("values")
        unconstrained_values = transform_fn(inv_transforms, init_values, invert=True)
        init_strategy = _init_to_unconstrained_value(values=unconstrained_values)
    prototype_params = transform_fn(inv_transforms, constrained_values, invert=True)
    (init_params, _, _), is_valid = find_valid_initial_params(
        rng_key,
        substitute(
            model,
            data={
                k: site["value"]
                for k, site in model_trace.items()
                if site["type"] in ["plate"]
            },
        ),
        init_strategy=init_strategy,
        enum=has_enumerate_support,
        model_args=model_args,
        model_kwargs=model_kwargs,
        prototype_params=prototype_params,
        forward_mode_differentiation=forward_mode_differentiation,
        validate_grad=False,
    )

    if not_jax_tracer(is_valid):
        if device_get(~jnp.all(is_valid)):
            with numpyro.validation_enabled(), trace() as tr:
                # validate parameters
                substituted_model(*model_args, **model_kwargs)
                # validate values
                for site in tr.values():
                    if site["type"] == "sample":
                        with warnings.catch_warnings(record=True) as ws:
                            site["fn"]._validate_sample(site["value"])
                        if len(ws) > 0:
                            for w in ws:
                                # at site information to the warning message
                                w.message.args = (
                                    "Site {}: {}".format(
                                        site["name"], w.message.args[0]
                                    ),
                                ) + w.message.args[1:]
                                warnings.showwarning(
                                    w.message,
                                    w.category,
                                    w.filename,
                                    w.lineno,
                                    file=w.file,
                                    line=w.line,
                                )
            raise RuntimeError(
                "Cannot find valid initial parameters. Please check your model again."
            )
    return ModelInfo(ParamInfo(init_params), model_trace)
