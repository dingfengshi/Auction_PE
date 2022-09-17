from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

import numpy as np
from base.base_generator import BaseGenerator


class Generator(BaseGenerator):
    def __init__(self, config, mode, X=None, ADV=None):
        super(Generator, self).__init__(config, mode)
        self.build_generator(X=X, ADV=ADV)

    def generate_random_X(self, shape):
        #         x_shape = [self.num_instances, self.num_agents, self.num_items]
        #         adv_shape = [self.num_misreports, self.num_instances, self.num_agents, self.num_items]
        num_agents = shape[1]
        num_items = shape[2]

        out = np.random.rand(*shape).astype(np.double)

        return out

    def generate_random_ADV(self, shape):
        return np.random.rand(*shape)
