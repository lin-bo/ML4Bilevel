#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

from abc import ABC, abstractmethod
from utils.functions import gen_betas


class AbstractSolver(ABC):

    def __init__(self, ins_name, save_model=False, potential='populations'):

        self.save_model = save_model
        self.ins_name = ins_name
        self.potential = potential

    @abstractmethod
    def solve(self, args, budget_proj, budget_sig, beta_1):
        """
        solve the problem
        :param args: the arguments of the instance (dict)
        :return:
        """
        raise NotImplementedError

    def _args2params(self, args):
        destinations = args['destinations']
        od_pairs = [(orig, des) for orig, destination in destinations.items() for des in destination]
        pop = args[self.potential]
        G = args['G']
        n_nodes = args['n_nodes'] if ('n_nodes' in args) else len(G.nodes())
        projs = args['projects']
        proj_costs = args['project_costs'] if ('project_costs' in args) else args['proj_costs']
        sig_costs = args['signal_costs']
        travel_time = args['travel_time']
        edge2proj = args['edge2proj']
        T = args['travel_time_limit']
        M = args['travel_time_max']
        return od_pairs, destinations, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T, M

    @staticmethod
    def _gen_betas(beta_1, T, M):
        return gen_betas(beta_1, T, M)
