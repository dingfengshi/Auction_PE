from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from base.base_generator import BaseGenerator
from scipy.stats import truncnorm


class Generator(BaseGenerator):
    def __init__(self, config, mode, X=None, ADV=None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X=X, ADV=ADV)

    def generate_random_X(self, shape):
        # x_shape = [self.num_instances, self.num_agents, self.num_items]
        # adv_shape = [self.num_misreports, self.num_instances, self.num_agents, self.num_items]
        mu, sigma = 0.3, 0.1
        out = truncnorm(a=-mu / sigma, b=(1 - mu) / sigma, loc=mu, scale=sigma).rvs(size=shape)

        return out

    def generate_random_ADV(self, shape):
        mu, sigma = 0.3, 0.1
        return truncnorm(a=-mu / sigma, b=(1 - mu) / sigma, loc=mu, scale=sigma).rvs(size=shape)
