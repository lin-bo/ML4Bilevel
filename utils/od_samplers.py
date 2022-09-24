#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

import numpy as np
from utils.functions import des2od


def naive_sampler(args, n, random_seed=None):
    # set random seed
    if random_seed:
        np.random.seed(random_seed)
    # retrieve od pairs
    destinations = args['destinations']
    od_pairs = des2od(destinations)
    if n > len(od_pairs):
        raise ValueError('n is greater than the number of od pairs')
    # sample
    indices = np.random.choice(range(len(od_pairs)), size=n, replace=False)
    return indices
