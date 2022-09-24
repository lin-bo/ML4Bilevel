#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

from abc import ABC, abstractmethod
from utils.functions import gen_betas


class AbstractSolver(ABC):

    def __init__(self, ins_name, save_model=False):

        self.save_model = save_model
        self.ins_name = ins_name

    @abstractmethod
    def solve(self, args, budget):
        """
        solve the problem
        :param args: the arguments of the instance (dict)
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def _args2params(args):
        destinations = args['destinations']
        if 'od_pairs' in args:
            od_pairs = args['od_pairs']
        else:
            od_pairs = [(orig, des) for orig, destination in destinations.items() for des in destination]
        pop = args['populations']
        seg2idx = args['seg2idx']
        v_bar = args['v_bar']
        beta = args['beta']
        G = args['G']
        n_nodes = args['n_nodes']
        projs = args['projects']
        proj_costs = args['project_costs']
        sig_costs = args['signal_costs']
        travel_time = args['travel_time']
        edge2proj = args['edge2proj']
        segidx2proj = args['segidx2proj']
        segs = args['segments']
        return od_pairs, destinations, pop, seg2idx, v_bar, beta, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, segidx2proj, segs

    @staticmethod
    def _gen_betas(beta_1, T, M):
        return gen_betas(beta_1, T, M)
