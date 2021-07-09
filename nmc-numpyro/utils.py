#!/usr/bin/env python3
# -*- coding: utf-8 -*-


def filter_unobserved_sites(modeltrace):
    return {
        name: site["value"]
        for name, site in modeltrace.items()
        if not site["is_observed"]
    }
