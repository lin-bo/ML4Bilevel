#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

from abc import ABC, abstractmethod
import numpy as np


class AbstractSolver(ABC):

    def __init__(self, ins_name, save_model=False):

        self.save_model = save_model
        self.ins_name = ins_name

    @abstractmethod
    def solve(self, args, budget_proj, budget_sig):
        """
        solve the problem
        :param args: the arguments of the instance (dict)
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def _args2params(args):
        destinations = args['destinations']
        od_pairs = [(orig, des) for orig, destination in destinations.items() for des in destination]
        pop = args['populations']
        G = args['G']
        n_nodes = args['n_nodes']
        projs = args['projects']
        proj_costs = args['project_costs']
        sig_costs = args['signal_costs']
        travel_time = args['travel_time']
        edge2proj = args['edge2proj']
        T = args['travel_time_limit']
        return od_pairs, destinations, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T
