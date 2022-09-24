#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

from solver.discrete.abstract_solver import AbstractSolver
import gurobipy as gp
from utils.functions import flatten
from utils.check import file_existence
import time
from tqdm import tqdm


def benders_cut(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        # get solution
        z = model.cbGetSolution(model._z)
        y = model.cbGetSolution(model._y)
        s = model.cbGetSolution(model._s)
        # get parameters
        edge2proj = model._edge2proj
        T = model._T
        # go through od pairs
        for (orig, des) in z:
            if z[orig, des] < 1 - 1e-5:
                continue
            # set an objective function for the sub-problem
            dual = model._sp[orig, des]
            obj = get_dual_obj(y, s, dual._lamb, dual._theta, dual._gamma, dual._tau, T, orig, des, edge2proj)
            dual.setObjective(obj, gp.GRB.MINIMIZE)
            dual.optimize()
            # generate cut
            if dual.status == gp.GRB.status.UNBOUNDED:
                lhs = generate_benders_lhs(orig, des, model, dual,
                                           list(dual._gamma.keys()), len(dual._lamb), len(dual._theta),
                                           list(dual._theta.keys()), edge2proj, T)
                model.cbLazy(lhs >= 0)


def get_dual_obj(y, s, lamb, theta, gamma, tau, T, orig, des, edge2proj):
    obj = lamb[des] - lamb[orig] + T * tau + \
          gp.quicksum(y[edge2proj[i, j]] * theta[i, j] for (i, j) in theta if y[edge2proj[i, j]] > 1 - 1e-5) + \
          gp.quicksum((y[edge2proj[g, h]] + s[i]) * gamma[i, j, g, h] for (i, j, g, h) in gamma if y[edge2proj[g, h]] + s[i] > 1 - 1e-5)
    return obj


def generate_benders_lhs(orig, des, model, dual, unsig_cross, n_lamb, n_theta, proj_edges, edge2proj, T):
    # get the extreme ray
    ray = dual.UnbdRay
    # cal coefficient for y and s
    y_coeffs = {idx: 0 for idx in model._y}
    s_coeffs = {idx: 0 for idx in model._s}
    for i, e in enumerate(proj_edges):
        if ray[i + n_lamb] > 1e-5:
            y_coeffs[edge2proj[e]] += ray[i + n_lamb]
    for j, crs in enumerate(unsig_cross):
        if ray[j + n_lamb + n_theta] > 1e-5:
            y_coeffs[edge2proj[crs[2], crs[3]]] += ray[j + n_lamb + n_theta]
            s_coeffs[crs[0]] += ray[j + n_lamb + n_theta]
    # generate lhs of the cut
    lhs = model._z[orig, des] * (ray[des] - ray[orig] + T * ray[-1]) + \
          gp.quicksum(model._y[i] * y_coeffs[i] for i in y_coeffs if y_coeffs[i] != 0) + \
          gp.quicksum(model._s[i] * s_coeffs[i] for i in s_coeffs if s_coeffs[i] != 0)
    # print(orig, des, ray[des], ray[orig], lhs)
    # print(lhs)
    return lhs


class BendersSolverFeasibilityCut(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, time_limit=None, regenerate=False):
        """
        solve the optimization problem
        :param args: instance arguments
        :param budget_proj:
        :param budget_sig:
        :param time_limit: solution time limit
        :return: lists of connected pairs, new projects, and new signals
        """
        print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T = self._args2params(args)
        print('The instance has {} od-pairs, {} projects, and {} nodes'.format(
            len(od_pairs), len(projs), n_nodes))
        print('compiling ...')
        tick = time.time()
        master = self._construct_master_problem(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig,
                                                proj_costs=proj_costs, sig_costs=sig_costs, time_limit=time_limit, regenerate=regenerate)
        master._sp = self._construct_subproblems(n_nodes, od_pairs, projs, sig_costs, G, travel_time, regenerate)
        master._edge2proj = edge2proj
        master._T = T
        print('  elapsed: {:.2f} sec'.format(time.time() - tick))
        print('solving ...')
        tick = time.time()
        master.optimize(benders_cut)
        print('  obj val: {:.2f}'.format(master.objVal))
        print('  elapsed: {:.2f} sec'.format(time.time() - tick))
        connected_pairs, new_projects, new_signals = self._get_solution(master)
        return connected_pairs, new_projects, new_signals

    def _construct_master_problem(self, od_pairs, pop, budget_proj, budget_sig, proj_costs, sig_costs, time_limit, regenerate):
        """
        construct the benders master problem
        :param od_pairs: a list of tuples representing the origin and destination of each pair
        :param pop: a list of population at each destination
        :param budget_proj: project budget
        :param budget_sig: signal budget
        :param proj_costs: dict of cost for each project
        :param sig_costs: dict of cost for each un-signalized node
        :param time_limit: solving time limit
        :return: a Gurobi Model
        """
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/discrete/master.mps'.format(self.ins_name)
        if file_existence(dir_name) and (not regenerate):
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_master_params(model, time_limit)
            z, y, s = self._get_master_variables(model, od_pairs, list(proj_costs.keys()), list(sig_costs.keys()))
            model = self._store_master_variables(model, z, y, s)
            model = self._update_master_rhs(model, budget_proj, budget_sig)
        else:
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_master_params(model, time_limit)
            # set parameters
            od_pops = {(orig, des): pop[des] for orig, des in od_pairs}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            # add variables
            z = model.addVars(od_pairs, name='z', vtype=gp.GRB.BINARY)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
            # add budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # set objective
            model.setObjective(z.prod(od_pops), gp.GRB.MAXIMIZE)
            # add variable dicts
            model = self._store_master_variables(model, z, y, s)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_master_params(model, time_limit):
        model.Params.outputFlag = 0
        if time_limit:
            model.Params.timeLimit = time_limit
        model.Params.lazyConstraints = 1
        return model

    @staticmethod
    def _get_master_variables(model, od_pairs, projects, signals):
        z = {(orig, des): model.getVarByName("z[{},{}]".format(orig, des)) for (orig, des) in od_pairs}
        y = {p: model.getVarByName("y[{}]".format(p)) for p in projects}
        s = {i: model.getVarByName("s[{}]".format(i)) for i in signals}
        return z, y, s

    @staticmethod
    def _store_master_variables(model, z, y, s):
        model._z = z
        model._y = y
        model._s = s
        return model

    @staticmethod
    def _update_master_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

    def _construct_subproblems(self, n_nodes, od_pairs, projs, sig_costs, G, travel_time, regenerate):
        """
        :param od_pairs: a list of tuples representing the origin and destination of each pair
        :param G: a NetworkX directed graph
        :return: a dict, key: od pair tuple, value: sub-problem
        """
        proj_edges = flatten(projs)
        proj_edge_set = self._proj_edge_set(proj_edges)
        unsig_set = set(sig_costs.keys())
        unsig_cross = [(i, j, g, h) for s in unsig_set for (i, j) in list(G.out_edges(s))
                       for (g, h) in list(G.in_edges(s)) + list(G.out_edges(s))
                       if self._diff_edges(i, j, g, h) and ('{}_{}'.format(g, h) in proj_edge_set)]
        gp.setParam('outputFlag', 0)
        probs = {}
        for orig, des in tqdm(od_pairs):
            probs[(orig, des)] = self._subproblem(n_nodes, orig, des, G, proj_edges, unsig_set, unsig_cross, travel_time, regenerate)
        gp.setParam('outputFlag', 1)
        return probs

    def _subproblem(self, n_nodes, orig, des, G, proj_edges, unsig_set, unsig_cross, travel_time, regenerate):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/discrete/sub_{}_{}.mps'.format(self.ins_name, orig, des)
        if file_existence(dir_name) and (not regenerate):
            model = gp.read(dir_name)
            model = self._set_sub_params(model)
            lamb, theta, gamma, tau = self._get_sub_variables(model, n_nodes, proj_edges, unsig_cross)
            model = self._store_sub_variables(model, lamb, theta, gamma, tau)
        else:
            # initialize the sub-problem
            model = gp.Model('subproblem_{}-{}'.format(orig, des))
            model = self._set_sub_params(model)
            # add variables
            lamb = model.addVars(n_nodes, name='lambda', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            theta = model.addVars(proj_edges, name='theta', lb=0, vtype=gp.GRB.CONTINUOUS)
            gamma = model.addVars(unsig_cross, name='gamma', lb=0, vtype=gp.GRB.CONTINUOUS)
            tau = model.addVar(name='tau', lb=0, vtype=gp.GRB.CONTINUOUS)
            # add constraints
            for i, j in G.edges:
                exp = lamb[j] - lamb[i] + travel_time[i, j] * tau
                if (i, j) in theta:
                    exp += theta[i, j]
                if i in unsig_set:
                    exp += gp.quicksum(gamma[i, j, g, h] for g, h in G.edges(i) if self._diff_edges(i, j, g, h) and (g, h) in theta)
                model.addConstr(exp >= 0, name='c')
            # store decision variables
            model = self._store_sub_variables(model, lamb, theta, gamma, tau)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_sub_params(model):
        model.Params.outputFlag = 0
        model.setParam('InfUnbdInfo', 1)
        return model

    @staticmethod
    def _get_sub_variables(model, n_nodes, proj_edges, unsig_cross):
        lamb = {i: model.getVarByName("lambda[{}]".format(i)) for i in range(n_nodes)}
        theta = {(i, j): model.getVarByName("theta[{},{}]".format(i, j)) for (i, j) in proj_edges}
        gamma = {(i, j, g, h): model.getVarByName("gamma[{},{},{},{}]".format(i, j, g, h)) for (i, j, g, h) in unsig_cross}
        tau = model.getVarByName("tau")
        return lamb, theta, gamma, tau

    @staticmethod
    def _store_sub_variables(model, lamb, theta, gamma, tau):
        model._lamb = lamb
        model._theta = theta
        model._gamma = gamma
        model._tau = tau
        return model

    @staticmethod
    def _diff_edges(i, j, g, h):
        """
        If edges (i, j) and (g, h) are identical
        :return: boolean
        """
        return i != g or j != h

    @staticmethod
    def _proj_edge_set(proj_edges):
        return set(['{}_{}'.format(i, j) for (i, j) in proj_edges])

    @staticmethod
    def _get_solution(model):
        """
        get the solution
        :param model: master problem
        :return: lists of connected pairs, new projects, new signals
        """
        z_val = model.getAttr('x', model._z)
        connected_pairs = [(orig, des) for (orig, des) in model._z if z_val[orig, des] >= 1 - 1e-5]
        y_val = model.getAttr('x', model._y)
        new_projects = [i for i in model._y if y_val[i] >= 1 - 1e-5]
        s_val = model.getAttr('x', model._s)
        new_signals = [i for i in model._s if s_val[i] >= 1 - 1e-5]
        return connected_pairs, new_projects, new_signals


class BendersSolverOptimalityCut(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, time_limit=None, regenerate=False):
        """
        solve the optimization problem
        :param args: instance arguments
        :param budget_proj:
        :param budget_sig:
        :param time_limit: solution time limit
        :return: lists of connected pairs, new projects, and new signals
        """
        print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T = self._args2params(args)
        print('The instance has {} od-pairs, {} projects, and {} nodes'.format(
            len(od_pairs), len(projs), n_nodes))
        print('compiling ...')
        tick = time.time()
        master = self._construct_master_problem(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig,
                                                proj_costs=proj_costs, sig_costs=sig_costs, time_limit=time_limit, regenerate=regenerate)
        master._sp = self._construct_subproblems(n_nodes, od_pairs, projs, sig_costs, G, travel_time, regenerate)
        master._edge2proj = edge2proj
        master._T = T
        print('  elapsed: {:.2f} sec'.format(time.time() - tick))
        print('solving ...')
        tick = time.time()
        master.optimize(benders_cut)
        print('  obj val: {:.2f}'.format(master.objVal))
        print('  elapsed: {:.2f} sec'.format(time.time() - tick))
        connected_pairs, new_projects, new_signals = self._get_solution(master)
        return connected_pairs, new_projects, new_signals

    def _construct_master_problem(self, od_pairs, pop, budget_proj, budget_sig, proj_costs, sig_costs, time_limit, regenerate):
        """
        construct the benders master problem
        :param od_pairs: a list of tuples representing the origin and destination of each pair
        :param pop: a list of population at each destination
        :param budget_proj: project budget
        :param budget_sig: signal budget
        :param proj_costs: dict of cost for each project
        :param sig_costs: dict of cost for each un-signalized node
        :param time_limit: solving time limit
        :return: a Gurobi Model
        """
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/master.mps'.format(self.ins_name)
        if file_existence(dir_name) and (not regenerate):
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_master_params(model, time_limit)
            z, y, s = self._get_master_variables(model, od_pairs, list(proj_costs.keys()), list(sig_costs.keys()))
            model = self._store_master_variables(model, z, y, s)
            model = self._update_master_rhs(model, budget_proj, budget_sig)
        else:
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_master_params(model, time_limit)
            # set parameters
            od_pops = {(orig, des): pop[des] for orig, des in od_pairs}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            # add variables
            z = model.addVars(od_pairs, name='z', vtype=gp.GRB.BINARY)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
            # add budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # set objective
            model.setObjective(z.prod(od_pops), gp.GRB.MAXIMIZE)
            # add variable dicts
            model = self._store_master_variables(model, z, y, s)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_master_params(model, time_limit):
        model.Params.outputFlag = 0
        if time_limit:
            model.Params.timeLimit = time_limit
        model.Params.lazyConstraints = 1
        return model

    @staticmethod
    def _get_master_variables(model, od_pairs, projects, signals):
        z = {(orig, des): model.getVarByName("z[{},{}]".format(orig, des)) for (orig, des) in od_pairs}
        y = {p: model.getVarByName("y[{}]".format(p)) for p in projects}
        s = {i: model.getVarByName("s[{}]".format(i)) for i in signals}
        return z, y, s

    @staticmethod
    def _store_master_variables(model, z, y, s):
        model._z = z
        model._y = y
        model._s = s
        return model

    @staticmethod
    def _update_master_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

    def _construct_subproblems(self, n_nodes, od_pairs, projs, sig_costs, G, travel_time, regenerate):
        """
        :param od_pairs: a list of tuples representing the origin and destination of each pair
        :param G: a NetworkX directed graph
        :return: a dict, key: od pair tuple, value: sub-problem
        """
        proj_edges = flatten(projs)
        proj_edge_set = self._proj_edge_set(proj_edges)
        unsig_set = set(sig_costs.keys())
        unsig_cross = [(i, j, g, h) for s in unsig_set for (i, j) in list(G.out_edges(s))
                       for (g, h) in list(G.in_edges(s)) + list(G.out_edges(s))
                       if self._diff_edges(i, j, g, h) and ('{}_{}'.format(g, h) in proj_edge_set)]
        gp.setParam('outputFlag', 0)
        probs = {}
        for orig, des in tqdm(od_pairs):
            probs[(orig, des)] = self._subproblem(n_nodes, orig, des, G, proj_edges, unsig_set, unsig_cross, travel_time, regenerate)
        gp.setParam('outputFlag', 1)
        return probs

    def _subproblem(self, n_nodes, orig, des, G, proj_edges, unsig_set, unsig_cross, travel_time, regenerate):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/sub_{}_{}.mps'.format(self.ins_name, orig, des)
        if file_existence(dir_name) and (not regenerate):
            model = gp.read(dir_name)
            model = self._set_sub_params(model)
            lamb, theta, gamma, tau = self._get_sub_variables(model, n_nodes, proj_edges, unsig_cross)
            model = self._store_sub_variables(model, lamb, theta, gamma, tau)
        else:
            # initialize the sub-problem
            model = gp.Model('subproblem_{}-{}'.format(orig, des))
            model = self._set_sub_params(model)
            # add variables
            lamb = model.addVars(n_nodes, name='lambda', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            theta = model.addVars(proj_edges, name='theta', lb=0, vtype=gp.GRB.CONTINUOUS)
            gamma = model.addVars(unsig_cross, name='gamma', lb=0, vtype=gp.GRB.CONTINUOUS)
            tau = model.addVar(name='tau', lb=0, vtype=gp.GRB.CONTINUOUS)
            # add constraints
            for i, j in G.edges:
                exp = lamb[j] - lamb[i] + travel_time[i, j] * tau
                if (i, j) in theta:
                    exp += theta[i, j]
                if i in unsig_set:
                    exp += gp.quicksum(gamma[i, j, g, h] for g, h in G.edges(i) if self._diff_edges(i, j, g, h) and (g, h) in theta)
                model.addConstr(exp >= 0, name='c')
            # store decision variables
            model = self._store_sub_variables(model, lamb, theta, gamma, tau)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_sub_params(model):
        model.Params.outputFlag = 0
        model.setParam('InfUnbdInfo', 1)
        return model

    @staticmethod
    def _get_sub_variables(model, n_nodes, proj_edges, unsig_cross):
        lamb = {i: model.getVarByName("lambda[{}]".format(i)) for i in range(n_nodes)}
        theta = {(i, j): model.getVarByName("theta[{},{}]".format(i, j)) for (i, j) in proj_edges}
        gamma = {(i, j, g, h): model.getVarByName("gamma[{},{},{},{}]".format(i, j, g, h)) for (i, j, g, h) in unsig_cross}
        tau = model.getVarByName("tau")
        return lamb, theta, gamma, tau

    @staticmethod
    def _store_sub_variables(model, lamb, theta, gamma, tau):
        model._lamb = lamb
        model._theta = theta
        model._gamma = gamma
        model._tau = tau
        return model

    @staticmethod
    def _diff_edges(i, j, g, h):
        """
        If edges (i, j) and (g, h) are identical
        :return: boolean
        """
        return i != g or j != h

    @staticmethod
    def _proj_edge_set(proj_edges):
        return set(['{}_{}'.format(i, j) for (i, j) in proj_edges])

    @staticmethod
    def _get_solution(model):
        """
        get the solution
        :param model: master problem
        :return: lists of connected pairs, new projects, new signals
        """
        z_val = model.getAttr('x', model._z)
        connected_pairs = [(orig, des) for (orig, des) in model._z if z_val[orig, des] >= 1 - 1e-5]
        y_val = model.getAttr('x', model._y)
        new_projects = [i for i in model._y if y_val[i] >= 1 - 1e-5]
        s_val = model.getAttr('x', model._s)
        new_signals = [i for i in model._s if s_val[i] >= 1 - 1e-5]
        return connected_pairs, new_projects, new_signals

