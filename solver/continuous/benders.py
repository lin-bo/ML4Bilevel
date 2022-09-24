#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

from solver.continuous.abstract_solver import AbstractSolver
import gurobipy as gp
from utils.functions import flatten, dump_file
from utils.check import file_existence
import time
from tqdm import tqdm
import numpy as np
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path_length
from utils.functions import load_file


def benders_cut(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        # get solution
        v = model.cbGetSolution(model._v)
        y = model.cbGetSolution(model._y)
        budget_sig = model._budget_sig
        s = model.cbGetSolution(model._s) if model._budget_sig > 0 else model._s.copy()
        # get parameters
        edge2proj = model._edge2proj
        M = model._M
        model._y_hat, model._s_hat = update_inner_points(model._y_hat, model._s_hat, y, s)
        # go through od pairs
        for (orig, des) in v:
            if v[orig, des] > M - 1e-12:
                continue
            # set an objective function for the sub-problem
            dual = model._sp[orig, des]
            obj = get_dual_obj(y, s, dual._lamb, dual._theta, dual._gamma, orig, des, edge2proj)
            dual.setObjective(obj, gp.GRB.MAXIMIZE)
            dual.optimize()
            # generate cut
            if model._pareto:
                rhs = pareto_rhs(model, dual, y, s, dual.objVal, orig, des, edge2proj, budget_sig)
            else:
                rhs = generate_benders_rhs(orig, des, model, dual, edge2proj, budget_sig)
            model.cbLazy(model._v[orig, des] >= rhs)
            model._cnt += 1


def benders_cut_root_relax(model, where):
    if where == gp.GRB.Callback.MIPSOL:
        # get solution
        v = model.cbGetSolution(model._v)
        y = model.cbGetSolution(model._y)
        budget_sig = model._budget_sig
        s = model.cbGetSolution(model._s) if model._budget_sig > 0 else model._s.copy()
        # get parameters
        edge2proj = model._edge2proj
        M = model._M
        # go through od pairs
        for (orig, des) in v:
            if v[orig, des] > M - 1e-12:
                continue
            # set an objective function for the sub-problem
            dual = model._sp[orig, des]
            obj = get_dual_obj(y, s, dual._lamb, dual._theta, dual._gamma, orig, des, edge2proj)
            dual.setObjective(obj, gp.GRB.MAXIMIZE)
            dual.optimize()
            # generate cut
            rhs, coeff = generate_benders_rhs(orig, des, model, dual, edge2proj, budget_sig, True)
            model.cbLazy(model._v[orig, des] >= rhs)
            model._cnt += 1
            model._coeffs.append(coeff)


def get_dual_obj(y, s, lamb, theta, gamma, orig, des, edge2proj):
    obj = - lamb[des] + lamb[orig] - \
          gp.quicksum(y[edge2proj[i, j]] * theta[i, j] for (i, j) in theta if y[edge2proj[i, j]] > 1e-8) - \
          gp.quicksum((y[edge2proj[g, h]] + s[i]) * gamma[i, j, g, h] for (i, j, g, h) in gamma if y[edge2proj[g, h]] + s[i] > 1e-8)
    return obj


def generate_benders_rhs(orig, des, model, dual, edge2proj, budget_sig, return_coeff=False):
    # get the solution
    lamb_val = dual.getAttr('x', dual._lamb)
    theta_val = dual.getAttr('x', dual._theta)
    gamma_val = dual.getAttr('x', dual._gamma)
    # set coefficients
    y_coeffs = {i: 0 for i in model._y}
    if budget_sig > 0:
        s_coeffs = {i: 0 for i in model._s}
    # consider theta
    for (i, j) in theta_val:
        if (i, j) in edge2proj and theta_val[i, j] > 1e-12:
                y_coeffs[edge2proj[i, j]] -= theta_val[i, j]
    # consider gamma
    for (i, j, g, h) in gamma_val:
        if gamma_val[i, j, g, h] > 1e-12:
            if (g, h) in edge2proj:
                y_coeffs[edge2proj[g, h]] -= gamma_val[i, j, g, h]
            if budget_sig > 0:
                if i in model._s:
                    s_coeffs[i] -= gamma_val[i, j, g, h]
    # calculate rhs
    rhs = - lamb_val[des] + lamb_val[orig] + \
          gp.quicksum(model._y[i] * y_coeffs[i] for i in model._y if y_coeffs[i] < -1e-12)
    if budget_sig > 0:
        rhs += gp.quicksum(model._s[i] * s_coeffs[i] for i in model._s if s_coeffs[i] < -1e-12)

    if not return_coeff:
        return rhs
    else:
        coeff = {'od': (orig, des),
                 'y': {idx: val for idx, val in y_coeffs.items() if val < -1e-12},
                 'c': - lamb_val[des] + lamb_val[orig]}
        if budget_sig > 0:
            coeff['s'] = {idx: val for idx, val in s_coeffs.items() if val < -1e-12}
        return rhs, coeff


def pareto_rhs(model, pareto_prob, y, s, obj_val, orig, des, edge2proj, budget_sig, return_coeff=False):
    obj_exp = get_dual_obj(y, s, pareto_prob._lamb, pareto_prob._theta, pareto_prob._gamma, orig, des, edge2proj)
    pareto_prob.addConstr(obj_exp >= obj_val, name='optimality')
    obj = get_dual_obj(model._y_hat, model._s_hat, pareto_prob._lamb, pareto_prob._theta, pareto_prob._gamma, orig, des, edge2proj)
    pareto_prob.setObjective(obj, gp.GRB.MAXIMIZE)
    pareto_prob.optimize()
    if return_coeff:
        rhs, coeff = generate_benders_rhs(orig, des, model, pareto_prob, edge2proj, budget_sig, return_coeff)
        pareto_prob.remove(pareto_prob.getConstrByName('optimality'))
        return rhs, coeff
    else:
        rhs = generate_benders_rhs(orig, des, model, pareto_prob, edge2proj, budget_sig, return_coeff)
        pareto_prob.remove(pareto_prob.getConstrByName('optimality'))
        return rhs


def update_inner_points(y_hat, s_hat, y, s):
    p = 0.5
    y_new = {idx: val * p + y[idx] * (1 - p) for idx, val in y_hat.items()}
    s_new = {idx: val * p + s[idx] * (1 - p) for idx, val in s_hat.items()}
    return y_new, s_new


class BendersSolverOptimalityCut(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, beta_1, quiet=False, fixed_project=[], ub=-100, manual_params = {},
              time_limit=None, regenerate=False, pareto=False, relax4cut=False, weighted=False, mip_gap=None):
        """
        solve the optimization problem
        :param args: instance arguments
        :param budget_proj:
        :param budget_sig:
        :param time_limit: solution time limit
        :return: lists of connected pairs, new projects, and new signals
        """
        if not quiet:
            print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T, M = self._args2params(args)
        weights = args['weights'] if 'weights' in args and weighted else {}
        beta = self._gen_betas(beta_1, T, M)
        if not quiet:
            print('The instance has {} od-pairs, {} projects, {} candiate intersections, {} edges, and {} nodes'.format(
                len(od_pairs), len(projs), len(sig_costs), len(G.edges()), n_nodes))
            print('Fixed project:', fixed_project)
            print('compiling ...')
        tick = time.time()
        if len(manual_params) > 0:
            T, M, beta = manual_params['T'], manual_params['M'], manual_params['beta']
        master = self._construct_master_problem(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig,
                                                T=T, beta=beta, weights=weights, proj_costs=proj_costs, sig_costs=sig_costs,
                                                time_limit=time_limit, regenerate=regenerate, quiet=quiet, mip_gap=mip_gap,
                                                fixed_project=fixed_project, ub=ub)
        if not quiet:
            master.update()
            print('Master problem has {} variables and {} constraints'.format(master.numVars, master.numConstrs))
        master._edge2proj = edge2proj
        master._T = T
        master._M = M
        master._y_hat, master._s_hat = self._find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs)
        master._sp = self._construct_subproblems(od_pairs, projs, sig_costs, G, travel_time, M, regenerate)
        if not quiet:
            master._sp[od_pairs[0]].update()
            print('A typical sub-problem problem has {} variables and {} constraints'.format(
                master._sp[od_pairs[0]].numVars, master._sp[od_pairs[0]].numConstrs))
        master._pareto = pareto
        master._cnt = 0
        if not quiet:
            print('  elapsed: {:.2f} sec'.format(time.time() - tick))
            print('solving ...')
        tick = time.time()
        if relax4cut:
            if not quiet:
                print('solving root relaxation for cheap cuts ...')
            cuts, v, y, s = self._relax4cut(master)
            cut_cnt = self._add_cheap_cuts(master, cuts, v, y, s)
            if not quiet:
                print('{} cuts found, {} of them added'.format(len(cuts), cut_cnt))
        n_cheap_cuts = cut_cnt if relax4cut else 0
        master.optimize(benders_cut)
        t_sol = time.time() - tick
        print('  obj val: {:.2f}'.format(master.objVal))
        print('  # of cuts added: {} = {} + {}'.format(n_cheap_cuts + master._cnt, n_cheap_cuts, master._cnt))
        print('  elapsed: {:.2f} sec'.format(t_sol))
        new_projects, new_signals = self._get_solution(master)
        return new_projects, new_signals, t_sol, master.objVal

    def _construct_master_problem(self, od_pairs, pop, budget_proj, budget_sig, proj_costs, sig_costs, beta, weights,
                                  T, time_limit, regenerate, quiet, mip_gap, fixed_project, ub):
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
        dir_name = './prob/{}/models/continuous/master.mps'.format(self.ins_name)
        if file_existence(dir_name) and (not regenerate):
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            u, v, y, s = self._get_master_variables(model, od_pairs, list(proj_costs.keys()), list(sig_costs.keys()), budget_sig)
            # store variables
            model._u = u
            model._v = v
            model._y = y
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in list(sig_costs.keys())}
            model._budget_sig = budget_sig
            model = self._update_master_rhs(model, budget_proj, budget_sig)
        else:
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            # set parameters
            obj_weights = weights if len(weights) > 0 else {(orig, des): pop[des] for orig, des in od_pairs}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            # add variables
            u = model.addVars(od_pairs, name='u', vtype=gp.GRB.CONTINUOUS)
            v = model.addVars(od_pairs, name='v', vtype=gp.GRB.CONTINUOUS)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            if budget_sig > 0:
                s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
            # add budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            model.addConstr(y[556] == 0)
            model.addConstr(y[1285] == 0)
            if budget_sig > 0:
                model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # fix some project if necessary
            if len(fixed_project):
                model.addConstrs((y[i] == 1 for i in fixed_project), name='fixed_project')
            # add time constraints
            model.addConstrs((u[orig, des] >= v[orig, des] - T for orig, des in od_pairs), name='time_exceeds')
            # set objective
            obj = beta[1] * v.prod(obj_weights) + (beta[2] - beta[1]) * u.prod(obj_weights)
            if ub > 0:
                model.addConstr(obj <= ub, name='ub_assist')
            model.setObjective(obj, gp.GRB.MINIMIZE)
            # add variable dicts
            model._u = u
            model._v = v
            model._y = y
            model._budget_sig = budget_sig
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in signals}
            # write the problem to local drive
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_master_params(model, time_limit, quiet, mip_gap):
        model.Params.outputFlag = 0 if quiet else 1
        if time_limit:
            model.Params.timeLimit = time_limit
        if mip_gap:
            model.Params.mipGap = mip_gap
        model.Params.lazyConstraints = 1
        # model.Params.presolve = 0
        return model

    @staticmethod
    def _get_master_variables(model, od_pairs, projects, signals, budget_sig):
        u = gp.tupledict([((orig, des), model.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        v = gp.tupledict([((orig, des), model.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        y = gp.tupledict([(i, model.getVarByName("y[{}]".format(i))) for i in projects])
        if budget_sig > 0:
            s = gp.tupledict([(i, model.getVarByName("s[{}]".format(i))) for i in signals])
        else:
            s = []
        return u, v, y, s

    @staticmethod
    def _update_master_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

    @staticmethod
    def _find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs):
        """
        find the raltiave inner points (for the pareto cut problem)
        :param budget_proj:
        :param budget_sig:
        :param proj_costs:
        :param sig_costs:
        :return: dict for y, dict for s
        """
        n_proj = len(proj_costs)
        n_sig = len(sig_costs)
        y_hat = {idx: np.min([1, budget_proj / (2 * n_proj * val)]) for idx, val in proj_costs.items()}
        if budget_sig > 0:
            s_hat = {idx: np.min([1, budget_sig / (2 * n_sig * val)]) for idx, val in sig_costs.items()}
        else:
            s_hat = {idx: 0 for idx, val in sig_costs.items()}
        return y_hat, s_hat

    def _relax4cut(self, model):
        # initialize the relaxed problem
        relaxed = self._gen_relaxed_problem(model)
        relaxed.Params.outputFlag = 0
        relaxed.optimize(benders_cut_root_relax)
        v, y, s = self._get_relaxed_sol(relaxed)
        return relaxed._coeffs, v, y, s

    @staticmethod
    def _gen_relaxed_problem(model):
        model.update()
        relaxed = model.relax()
        relaxed._u = gp.tupledict([((orig, des), relaxed.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in model._u])
        relaxed._v = gp.tupledict([((orig, des), relaxed.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in model._v])
        relaxed._y = gp.tupledict([(i, relaxed.getVarByName("y[{}]".format(i))) for i in model._y])
        if model._budget_sig > 0:
            relaxed._s = gp.tupledict([(i, relaxed.getVarByName("s[{}]".format(i))) for i in model._s])
        else:
            relaxed._s = model._s.copy()
        # store parameters
        relaxed._M = model._M
        relaxed._edge2proj = model._edge2proj
        relaxed._cnt = 0
        relaxed._sp = model._sp
        relaxed._coeffs = []
        relaxed._y_hat = model._y_hat
        relaxed._s_hat = model._s_hat
        relaxed._budget_sig = model._budget_sig
        # add the dummy variable and the dummy constraint
        d = relaxed.addVar(name='dummy', vtype=gp.GRB.BINARY)
        relaxed.addConstr(d == 0, name='dummy')
        return relaxed

    @staticmethod
    def _get_relaxed_sol(model):
        if model.status == 9:
            return {}, {}, {}
        v = model.getAttr('x', model._v)
        y = model.getAttr('x', model._y)
        s = model.getAttr('x', model._s) if model._budget_sig > 0 else model._s.copy()
        return v, y, s

    def _add_cheap_cuts(self, model, cuts, v, y, s):
        cnt = 0
        not_opt = len(y) == 0
        for idx, cut in enumerate(cuts):
            if not_opt or (self._binding(cut, v, y, s)):
                cnt += 1
                rhs = cut['c'] + model._y.prod(cut['y'])
                if 's' in cut:
                    rhs += model._s.prod(cut['s'])
                model.addConstr(model._v[cut['od']] >= rhs, name='cheap{}'.format(idx))
        return cnt

    def _binding(self, cut, v, y, s):
        y_prod = np.sum([y[idx] * val for idx, val in cut['y'].items()])
        s_prod = np.sum([s[idx] * val for idx, val in cut['s'].items()]) if 's' in cut else 0
        rhs = cut['c'] + y_prod + s_prod
        return (v[cut['od']] > rhs - 1e-9) and (v[cut['od']] < rhs + 1e-9)

    def _construct_subproblems(self, od_pairs, projs, sig_costs, G, travel_time, M, regenerate):
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
        G_reverse = G.reverse()
        probs = {}
        for orig, des in tqdm(od_pairs):
            probs[(orig, des)] = self._subproblem(orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate)
        return probs

    def _subproblem(self, orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/continuous/sub_{}_{}.mps'.format(self.ins_name, orig, des)
        if file_existence(dir_name) and (not regenerate):
            model = gp.read(dir_name)
            model = self._set_sub_params(model)
            lamb, theta, gamma = self._get_sub_variables(model, proj_edges, unsig_cross, G)
            model = self._store_sub_variables(model, lamb, theta, gamma)
        else:
            # initialize the sub-problem
            model = gp.Model('subproblem_{}-{}'.format(orig, des))
            model = self._set_sub_params(model)
            # parameters
            relevant_nodes, relevant_edges = self._get_relevant_nodes_edges(G, G_reverse, orig, des, M)
            proj_edges = set(proj_edges)
            theta_edges = [(i, j) for i, j in relevant_edges if (i, j) in proj_edges]
            relevant_edges = set(relevant_edges)
            gamma_unsig = [(i, j, g, h) for (i, j, g, h) in unsig_cross if (i, j) in relevant_edges]
            relevant_edges = list(relevant_edges)
            # add variables
            lamb = model.addVars(relevant_nodes, name='lambda', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            theta = model.addVars(theta_edges, name='theta', lb=0, vtype=gp.GRB.CONTINUOUS)
            gamma = model.addVars(gamma_unsig, name='gamma', lb=0, vtype=gp.GRB.CONTINUOUS)
            # add constraints - x
            for i, j in relevant_edges:
                exp = - lamb[j] + lamb[i]
                if (i, j) in theta:
                    exp -= theta[i, j]
                if i in unsig_set:
                    exp -= gp.quicksum(gamma[i, j, g, h] for g, h in G.edges(i) if self._diff_edges(i, j, g, h) and (g, h) in theta)
                model.addConstr(exp <= travel_time[i, j], name='x')
            # add constraints - z
            model.addConstr(-lamb[des] + lamb[orig] <= M, name='f')
            # store decision variables
            model = self._store_sub_variables(model, lamb, theta, gamma)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_sub_params(model):
        model.Params.outputFlag = 0
        return model

    @staticmethod
    def _get_sub_variables(model, proj_edges, unsig_cross, G):
        lamb = {i: model.getVarByName("lambda[{}]".format(i)) for i in G.nodes}
        theta = {(i, j): model.getVarByName("theta[{},{}]".format(i, j)) for (i, j) in proj_edges}
        gamma = {(i, j, g, h): model.getVarByName("gamma[{},{},{},{}]".format(i, j, g, h)) for (i, j, g, h) in unsig_cross}
        return lamb, theta, gamma

    @staticmethod
    def _store_sub_variables(model, lamb, theta, gamma):
        model._lamb = lamb
        model._theta = theta
        model._gamma = gamma
        return model

    @staticmethod
    def _get_relevant_nodes_edges(G, G_reverse, orig, des, M):
        reachable_orig = single_source_dijkstra_path_length(G=G, source=orig, cutoff=M, weight='time')
        reachable_des = single_source_dijkstra_path_length(G=G_reverse, source=des, cutoff=M, weight='time')
        relevant_nodes = set([])
        for node, val in reachable_orig.items():
            if (node in reachable_des) and (reachable_des[node] + val <= M):
                relevant_nodes.add(node)
                for (node, tnode) in G.out_edges(node):
                    relevant_nodes.add(tnode)
                for (fnode, node) in G.in_edges(node):
                    relevant_nodes.add(fnode)
        relevant_edges = []
        for fnode, tnode in G.edges():
            if (fnode in relevant_nodes) and (tnode in relevant_nodes):
                relevant_edges.append((fnode, tnode))
        return relevant_nodes, relevant_edges

    def _gen_pareto_problems(self, duals, n_nodes, projs, sig_costs, G):
        proj_edges = flatten(projs)
        proj_edge_set = self._proj_edge_set(proj_edges)
        unsig_set = set(sig_costs.keys())
        unsig_cross = [(i, j, g, h) for s in unsig_set for (i, j) in list(G.out_edges(s))
                       for (g, h) in list(G.in_edges(s)) + list(G.out_edges(s))
                       if self._diff_edges(i, j, g, h) and ('{}_{}'.format(g, h) in proj_edge_set)]
        print('generating pareto problem ...')
        probs = {}
        for orig, des in tqdm(duals):
            probs[orig, des] = self._pareto_problem(duals[orig, des], n_nodes, proj_edges, unsig_cross)
        return probs

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
        # penalties = model.getAttr('x', model._p)
        y_val = model.getAttr('x', model._y)
        new_projects = [i for i in model._y if y_val[i] >= 1 - 1e-5]
        if model._budget_sig > 0:
            s_val = model.getAttr('x', model._s)
            new_signals = [i for i in model._s if s_val[i] >= 1 - 1e-5]
        else:
            new_signals = []
        return new_projects, new_signals


class BendersSolverOptimalityCutRegu(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, beta_1, quiet=False, fixed_project=[], ub=-100, manual_params = {},
              time_limit=None, regenerate=False, pareto=False, relax4cut=False, weighted=False, mip_gap=None, regu_param=1):
        """
        solve the optimization problem
        :param args: instance arguments
        :param budget_proj:
        :param budget_sig:
        :param time_limit: solution time limit
        :return: lists of connected pairs, new projects, and new signals
        """
        if not quiet:
            print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T, M = self._args2params(args)
        weights = args['weights'] if 'weights' in args and weighted else {}
        beta = self._gen_betas(beta_1, T, M)
        if not quiet:
            print('The instance has {} od-pairs, {} projects, {} candiate intersections, {} edges, and {} nodes'.format(
                len(od_pairs), len(projs), len(sig_costs), len(G.edges()), n_nodes))
            print('Fixed project:', fixed_project)
            print('compiling ...')
        tick = time.time()
        if len(manual_params) > 0:
            T, M, beta = manual_params['T'], manual_params['M'], manual_params['beta']
        master = self._construct_master_problem(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig,
                                                T=T, beta=beta, weights=weights, proj_costs=proj_costs, sig_costs=sig_costs,
                                                time_limit=time_limit, regenerate=regenerate, quiet=quiet, mip_gap=mip_gap,
                                                fixed_project=fixed_project, ub=ub, regu_param=regu_param)
        if not quiet:
            master.update()
            print('Master problem has {} variables and {} constraints'.format(master.numVars, master.numConstrs))
        master._edge2proj = edge2proj
        master._T = T
        master._M = M
        master._y_hat, master._s_hat = self._find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs)
        master._sp = self._construct_subproblems(od_pairs, projs, sig_costs, G, travel_time, M, regenerate)
        if not quiet:
            master._sp[od_pairs[0]].update()
            print('A typical sub-problem problem has {} variables and {} constraints'.format(
                master._sp[od_pairs[0]].numVars, master._sp[od_pairs[0]].numConstrs))
        master._pareto = pareto
        master._cnt = 0
        if not quiet:
            print('  elapsed: {:.2f} sec'.format(time.time() - tick))
            print('solving ...')
        tick = time.time()
        if relax4cut:
            if not quiet:
                print('solving root relaxation for cheap cuts ...')
            cuts, v, y, s = self._relax4cut(master)
            cut_cnt = self._add_cheap_cuts(master, cuts, v, y, s)
            if not quiet:
                print('{} cuts found, {} of them added'.format(len(cuts), cut_cnt))
        n_cheap_cuts = cut_cnt if relax4cut else 0
        master.optimize(benders_cut)
        t_sol = time.time() - tick
        print('  obj val: {:.2f}'.format(master.objVal))
        print('  # of cuts added: {} = {} + {}'.format(n_cheap_cuts + master._cnt, n_cheap_cuts, master._cnt))
        print('  elapsed: {:.2f} sec'.format(t_sol))
        new_projects, new_signals = self._get_solution(master)
        return new_projects, new_signals, t_sol, master.objVal

    def _construct_master_problem(self, od_pairs, pop, budget_proj, budget_sig, proj_costs, sig_costs, beta, weights,
                                  T, time_limit, regenerate, quiet, mip_gap, fixed_project, ub, regu_param):
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
        dir_name = './prob/{}/models/continuous/master.mps'.format(self.ins_name)
        if file_existence(dir_name) and (not regenerate):
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            u, v, y, s = self._get_master_variables(model, od_pairs, list(proj_costs.keys()), list(sig_costs.keys()), budget_sig)
            # store variables
            model._u = u
            model._v = v
            model._y = y
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in list(sig_costs.keys())}
            model._budget_sig = budget_sig
            model = self._update_master_rhs(model, budget_proj, budget_sig)
        else:
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            # set parameters
            obj_weights = weights if len(weights) > 0 else {(orig, des): pop[des] for orig, des in od_pairs}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            # add variables
            u = model.addVars(od_pairs, name='u', vtype=gp.GRB.CONTINUOUS)
            v = model.addVars(od_pairs, name='v', vtype=gp.GRB.CONTINUOUS)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            if budget_sig > 0:
                s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
            # add budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            if budget_sig > 0:
                model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # fix some project if necessary
            if len(fixed_project):
                model.addConstrs((y[i] == 1 for i in fixed_project), name='fixed_project')
            # add time constraints
            model.addConstrs((u[orig, des] >= v[orig, des] - T for orig, des in od_pairs), name='time_exceeds')
            # set objective
            tot_weight = np.sum(list(obj_weights.values()))
            obj = beta[1] * v.prod(obj_weights) + (beta[2] - beta[1]) * u.prod(obj_weights) + \
                  regu_param * tot_weight * y.sum()
            if ub > 0:
                model.addConstr(obj <= ub, name='ub_assist')
            model.setObjective(obj, gp.GRB.MINIMIZE)
            # add variable dicts
            model._u = u
            model._v = v
            model._y = y
            model._budget_sig = budget_sig
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in signals}
            # write the problem to local drive
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_master_params(model, time_limit, quiet, mip_gap):
        model.Params.outputFlag = 0 if quiet else 1
        if time_limit:
            model.Params.timeLimit = time_limit
        if mip_gap:
            model.Params.mipGap = mip_gap
        model.Params.lazyConstraints = 1
        # model.Params.presolve = 0
        return model

    @staticmethod
    def _get_master_variables(model, od_pairs, projects, signals, budget_sig):
        u = gp.tupledict([((orig, des), model.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        v = gp.tupledict([((orig, des), model.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        y = gp.tupledict([(i, model.getVarByName("y[{}]".format(i))) for i in projects])
        if budget_sig > 0:
            s = gp.tupledict([(i, model.getVarByName("s[{}]".format(i))) for i in signals])
        else:
            s = []
        return u, v, y, s

    @staticmethod
    def _update_master_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

    @staticmethod
    def _find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs):
        """
        find the raltiave inner points (for the pareto cut problem)
        :param budget_proj:
        :param budget_sig:
        :param proj_costs:
        :param sig_costs:
        :return: dict for y, dict for s
        """
        n_proj = len(proj_costs)
        n_sig = len(sig_costs)
        y_hat = {idx: np.min([1, budget_proj / (2 * n_proj * val)]) for idx, val in proj_costs.items()}
        if budget_sig > 0:
            s_hat = {idx: np.min([1, budget_sig / (2 * n_sig * val)]) for idx, val in sig_costs.items()}
        else:
            s_hat = {idx: 0 for idx, val in sig_costs.items()}
        return y_hat, s_hat

    def _relax4cut(self, model):
        # initialize the relaxed problem
        relaxed = self._gen_relaxed_problem(model)
        relaxed.Params.outputFlag = 0
        relaxed.optimize(benders_cut_root_relax)
        v, y, s = self._get_relaxed_sol(relaxed)
        return relaxed._coeffs, v, y, s

    @staticmethod
    def _gen_relaxed_problem(model):
        model.update()
        relaxed = model.relax()
        relaxed.Params.timeLimit = 1800
        relaxed._u = gp.tupledict([((orig, des), relaxed.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in model._u])
        relaxed._v = gp.tupledict([((orig, des), relaxed.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in model._v])
        relaxed._y = gp.tupledict([(i, relaxed.getVarByName("y[{}]".format(i))) for i in model._y])
        if model._budget_sig > 0:
            relaxed._s = gp.tupledict([(i, relaxed.getVarByName("s[{}]".format(i))) for i in model._s])
        else:
            relaxed._s = model._s.copy()
        # store parameters
        relaxed._M = model._M
        relaxed._edge2proj = model._edge2proj
        relaxed._cnt = 0
        relaxed._sp = model._sp
        relaxed._coeffs = []
        relaxed._y_hat = model._y_hat
        relaxed._s_hat = model._s_hat
        relaxed._budget_sig = model._budget_sig
        # add the dummy variable and the dummy constraint
        d = relaxed.addVar(name='dummy', vtype=gp.GRB.BINARY)
        relaxed.addConstr(d == 0, name='dummy')
        return relaxed

    @staticmethod
    def _get_relaxed_sol(model):
        if model.status == 9:
            return {}, {}, {}
        v = model.getAttr('x', model._v)
        y = model.getAttr('x', model._y)
        s = model.getAttr('x', model._s) if model._budget_sig > 0 else model._s.copy()
        return v, y, s

    def _add_cheap_cuts(self, model, cuts, v, y, s):
        cnt = 0
        not_opt = len(y) == 0
        for idx, cut in enumerate(cuts):
            if not_opt or (self._binding(cut, v, y, s)):
                cnt += 1
                rhs = cut['c'] + model._y.prod(cut['y'])
                if 's' in cut:
                    rhs += model._s.prod(cut['s'])
                model.addConstr(model._v[cut['od']] >= rhs, name='cheap{}'.format(idx))
        return cnt

    def _binding(self, cut, v, y, s):
        y_prod = np.sum([y[idx] * val for idx, val in cut['y'].items()])
        s_prod = np.sum([s[idx] * val for idx, val in cut['s'].items()]) if 's' in cut else 0
        rhs = cut['c'] + y_prod + s_prod
        return (v[cut['od']] > rhs - 1e-9) and (v[cut['od']] < rhs + 1e-9)

    def _construct_subproblems(self, od_pairs, projs, sig_costs, G, travel_time, M, regenerate):
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
        G_reverse = G.reverse()
        probs = {}
        for orig, des in tqdm(od_pairs):
            probs[(orig, des)] = self._subproblem(orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate)
        return probs

    def _subproblem(self, orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/continuous/sub_{}_{}.mps'.format(self.ins_name, orig, des)
        if file_existence(dir_name) and (not regenerate):
            model = gp.read(dir_name)
            model = self._set_sub_params(model)
            lamb, theta, gamma = self._get_sub_variables(model, proj_edges, unsig_cross, G)
            model = self._store_sub_variables(model, lamb, theta, gamma)
        else:
            # initialize the sub-problem
            model = gp.Model('subproblem_{}-{}'.format(orig, des))
            model = self._set_sub_params(model)
            # parameters
            relevant_nodes, relevant_edges = self._get_relevant_nodes_edges(G, G_reverse, orig, des, M)
            proj_edges = set(proj_edges)
            theta_edges = [(i, j) for i, j in relevant_edges if (i, j) in proj_edges]
            relevant_edges = set(relevant_edges)
            gamma_unsig = [(i, j, g, h) for (i, j, g, h) in unsig_cross if (i, j) in relevant_edges]
            relevant_edges = list(relevant_edges)
            # add variables
            lamb = model.addVars(relevant_nodes, name='lambda', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            theta = model.addVars(theta_edges, name='theta', lb=0, vtype=gp.GRB.CONTINUOUS)
            gamma = model.addVars(gamma_unsig, name='gamma', lb=0, vtype=gp.GRB.CONTINUOUS)
            # add constraints - x
            for i, j in relevant_edges:
                exp = - lamb[j] + lamb[i]
                if (i, j) in theta:
                    exp -= theta[i, j]
                if i in unsig_set:
                    exp -= gp.quicksum(gamma[i, j, g, h] for g, h in G.edges(i) if self._diff_edges(i, j, g, h) and (g, h) in theta)
                model.addConstr(exp <= travel_time[i, j], name='x')
            # add constraints - z
            model.addConstr(-lamb[des] + lamb[orig] <= M, name='f')
            # store decision variables
            model = self._store_sub_variables(model, lamb, theta, gamma)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_sub_params(model):
        model.Params.outputFlag = 0
        return model

    @staticmethod
    def _get_sub_variables(model, proj_edges, unsig_cross, G):
        lamb = {i: model.getVarByName("lambda[{}]".format(i)) for i in G.nodes}
        theta = {(i, j): model.getVarByName("theta[{},{}]".format(i, j)) for (i, j) in proj_edges}
        gamma = {(i, j, g, h): model.getVarByName("gamma[{},{},{},{}]".format(i, j, g, h)) for (i, j, g, h) in unsig_cross}
        return lamb, theta, gamma

    @staticmethod
    def _store_sub_variables(model, lamb, theta, gamma):
        model._lamb = lamb
        model._theta = theta
        model._gamma = gamma
        return model

    @staticmethod
    def _get_relevant_nodes_edges(G, G_reverse, orig, des, M):
        reachable_orig = single_source_dijkstra_path_length(G=G, source=orig, cutoff=M, weight='time')
        reachable_des = single_source_dijkstra_path_length(G=G_reverse, source=des, cutoff=M, weight='time')
        relevant_nodes = set([])
        for node, val in reachable_orig.items():
            if (node in reachable_des) and (reachable_des[node] + val <= M):
                relevant_nodes.add(node)
                for (node, tnode) in G.out_edges(node):
                    relevant_nodes.add(tnode)
                for (fnode, node) in G.in_edges(node):
                    relevant_nodes.add(fnode)
        relevant_edges = []
        for fnode, tnode in G.edges():
            if (fnode in relevant_nodes) and (tnode in relevant_nodes):
                relevant_edges.append((fnode, tnode))
        return relevant_nodes, relevant_edges

    def _gen_pareto_problems(self, duals, n_nodes, projs, sig_costs, G):
        proj_edges = flatten(projs)
        proj_edge_set = self._proj_edge_set(proj_edges)
        unsig_set = set(sig_costs.keys())
        unsig_cross = [(i, j, g, h) for s in unsig_set for (i, j) in list(G.out_edges(s))
                       for (g, h) in list(G.in_edges(s)) + list(G.out_edges(s))
                       if self._diff_edges(i, j, g, h) and ('{}_{}'.format(g, h) in proj_edge_set)]
        print('generating pareto problem ...')
        probs = {}
        for orig, des in tqdm(duals):
            probs[orig, des] = self._pareto_problem(duals[orig, des], n_nodes, proj_edges, unsig_cross)
        return probs

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
        # penalties = model.getAttr('x', model._p)
        y_val = model.getAttr('x', model._y)
        new_projects = [i for i in model._y if y_val[i] >= 1 - 1e-5]
        if model._budget_sig > 0:
            s_val = model.getAttr('x', model._s)
            new_signals = [i for i in model._s if s_val[i] >= 1 - 1e-5]
        else:
            new_signals = []
        return new_projects, new_signals


class BendersSolverOptimalityCutVariant(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, beta_1, quiet=False, fixed_project=[], ub=-100, manual_params = {},
              time_limit=None, regenerate=False, pareto=False, relax4cut=False, weighted=False, mip_gap=None):
        """
        solve the optimization problem
        :param args: instance arguments
        :param budget_proj:
        :param budget_sig:
        :param time_limit: solution time limit
        :return: lists of connected pairs, new projects, and new signals
        """
        if not quiet:
            print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T, M = self._args2params(args)
        weights = args['weights'] if 'weights' in args and weighted else {}
        beta = self._gen_betas(beta_1, T, M)
        if not quiet:
            print('The instance has {} od-pairs, {} projects, {} candiate intersections, {} edges, and {} nodes'.format(
                len(od_pairs), len(projs), len(sig_costs), len(G.edges()), n_nodes))
            print('Fixed project:', fixed_project)
            print('compiling ...')
        tick = time.time()
        if len(manual_params) > 0:
            T, M, beta = manual_params['T'], manual_params['M'], manual_params['beta']
        master = self._construct_master_problem(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig,
                                                T=T, M=M, beta=beta, weights=weights, proj_costs=proj_costs, sig_costs=sig_costs,
                                                time_limit=time_limit, regenerate=regenerate, quiet=quiet, mip_gap=mip_gap,
                                                fixed_project=fixed_project, ub=ub)
        if not quiet:
            master.update()
            print('Master problem has {} variables and {} constraints'.format(master.numVars, master.numConstrs))
        master._edge2proj = edge2proj
        master._T = T
        master._M = M
        master._y_hat, master._s_hat = self._find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs)
        master._sp = self._construct_subproblems(od_pairs, projs, sig_costs, G, travel_time, M, regenerate)
        if not quiet:
            master._sp[od_pairs[0]].update()
            print('A typical sub-problem problem has {} variables and {} constraints'.format(
                master._sp[od_pairs[0]].numVars, master._sp[od_pairs[0]].numConstrs))
        master._pareto = pareto
        master._cnt = 0
        if not quiet:
            print('  elapsed: {:.2f} sec'.format(time.time() - tick))
            print('solving ...')
        tick = time.time()
        if relax4cut:
            if not quiet:
                print('solving root relaxation for cheap cuts ...')
            cuts, v, y, s = self._relax4cut(master)
            cut_cnt = self._add_cheap_cuts(master, cuts, v, y, s)
            if not quiet:
                print('{} cuts found, {} of them added'.format(len(cuts), cut_cnt))
        n_cheap_cuts = cut_cnt if relax4cut else 0
        master.optimize(benders_cut)
        t_sol = time.time() - tick
        print('  obj val: {:.2f}'.format(master.objVal))
        print('  # of cuts added: {} = {} + {}'.format(n_cheap_cuts + master._cnt, n_cheap_cuts, master._cnt))
        print('  elapsed: {:.2f} sec'.format(t_sol))
        new_projects, new_signals = self._get_solution(master)
        return new_projects, new_signals, t_sol, master.objVal

    def _construct_master_problem(self, od_pairs, pop, budget_proj, budget_sig, proj_costs, sig_costs, beta, weights,
                                  T, M, time_limit, regenerate, quiet, mip_gap, fixed_project, ub):
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
        dir_name = './prob/{}/models/continuous/master.mps'.format(self.ins_name)
        if file_existence(dir_name) and (not regenerate):
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            u, v, y, s = self._get_master_variables(model, od_pairs, list(proj_costs.keys()), list(sig_costs.keys()), budget_sig)
            # store variables
            model._u = u
            model._v = v
            model._y = y
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in list(sig_costs.keys())}
            model._budget_sig = budget_sig
            model = self._update_master_rhs(model, budget_proj, budget_sig)
        else:
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            # set parameters
            obj_weights = weights if len(weights) > 0 else {(orig, des): pop[des] for orig, des in od_pairs}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            convex = beta[1] >= beta[2]
            # add variables
            v = model.addVars(od_pairs, name='v', vtype=gp.GRB.CONTINUOUS)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            if convex:
                x = model.addVars(od_pairs, name='x', ub=1, lb=0, vtype=gp.GRB.CONTINUOUS)
                w = model.addVars(od_pairs, 2, name='w', vtype=gp.GRB.CONTINUOUS)
            else:
                u = model.addVars(od_pairs, name='u', vtype=gp.GRB.CONTINUOUS)
            if budget_sig > 0:
                s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
            # add budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            if budget_sig > 0:
                model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # fix some project if necessary
            if len(fixed_project):
                model.addConstrs((y[i] == 1 for i in fixed_project), name='fixed_project')
            if convex:
                model.addConstrs((w[orig, des, 0] <= T * x[orig, des] for orig, des in od_pairs), name='interval_0_leq')
                model.addConstrs((w[orig, des, 1] <= M * (1 - x[orig, des])
                                  for orig, des in od_pairs), name='interval_1_leq')
                model.addConstrs((w[orig, des, 1] >= T * (1 - x[orig, des])
                                  for orig, des in od_pairs), name='interval_1_geq')
                model.addConstrs((w[orig, des, 0] + w[orig, des, 1] == v[orig, des]
                                  for orig, des in od_pairs), name='sum_w')
                # set obj
                intercept = {0: beta[0], 1: beta[0] - beta[1] * T + beta[2] * T}
                slope = {0: beta[1], 1: beta[2]}
                obj = gp.quicksum((- intercept[idx] * x[orig, des] + slope[idx] * w[orig, des, idx]) * obj_weights[orig, des]
                                  for (orig, des) in od_pairs for idx in range(2))
            else:
                # add time constraints
                model.addConstrs((u[orig, des] >= v[orig, des] - T for orig, des in od_pairs), name='time_exceeds')
                # set objective
                obj = beta[1] * v.prod(obj_weights) + (beta[2] - beta[1]) * u.prod(obj_weights)
            if ub > 0:
                model.addConstr(obj <= ub, name='ub_assist')
            model.setObjective(obj, gp.GRB.MINIMIZE)
            # add variable dicts
            if not convex:
                model._u = u
            model._v = v
            model._y = y
            model._budget_sig = budget_sig
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in signals}
            # write the problem to local drive
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_master_params(model, time_limit, quiet, mip_gap):
        model.Params.outputFlag = 0 if quiet else 1
        if time_limit:
            model.Params.timeLimit = time_limit
        if mip_gap:
            model.Params.mipGap = mip_gap
        model.Params.lazyConstraints = 1
        # model.Params.presolve = 0
        return model

    @staticmethod
    def _get_master_variables(model, od_pairs, projects, signals, budget_sig):
        u = gp.tupledict([((orig, des), model.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        v = gp.tupledict([((orig, des), model.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        y = gp.tupledict([(i, model.getVarByName("y[{}]".format(i))) for i in projects])
        if budget_sig > 0:
            s = gp.tupledict([(i, model.getVarByName("s[{}]".format(i))) for i in signals])
        else:
            s = []
        return u, v, y, s

    @staticmethod
    def _update_master_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

    @staticmethod
    def _find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs):
        """
        find the raltiave inner points (for the pareto cut problem)
        :param budget_proj:
        :param budget_sig:
        :param proj_costs:
        :param sig_costs:
        :return: dict for y, dict for s
        """
        n_proj = len(proj_costs)
        n_sig = len(sig_costs)
        y_hat = {idx: np.min([1, budget_proj / (2 * n_proj * val)]) for idx, val in proj_costs.items()}
        if budget_sig > 0:
            s_hat = {idx: np.min([1, budget_sig / (2 * n_sig * val)]) for idx, val in sig_costs.items()}
        else:
            s_hat = {idx: 0 for idx, val in sig_costs.items()}
        return y_hat, s_hat

    def _relax4cut(self, model):
        # initialize the relaxed problem
        relaxed = self._gen_relaxed_problem(model)
        relaxed.Params.outputFlag = 0
        relaxed.optimize(benders_cut_root_relax)
        v, y, s = self._get_relaxed_sol(relaxed)
        return relaxed._coeffs, v, y, s

    @staticmethod
    def _gen_relaxed_problem(model):
        model.update()
        relaxed = model.relax()
        # relaxed._u = gp.tupledict([((orig, des), relaxed.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in model._u])
        relaxed._v = gp.tupledict([((orig, des), relaxed.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in model._v])
        relaxed._y = gp.tupledict([(i, relaxed.getVarByName("y[{}]".format(i))) for i in model._y])
        if model._budget_sig > 0:
            relaxed._s = gp.tupledict([(i, relaxed.getVarByName("s[{}]".format(i))) for i in model._s])
        else:
            relaxed._s = model._s.copy()
        # store parameters
        relaxed._M = model._M
        relaxed._edge2proj = model._edge2proj
        relaxed._cnt = 0
        relaxed._sp = model._sp
        relaxed._coeffs = []
        relaxed._y_hat = model._y_hat
        relaxed._s_hat = model._s_hat
        relaxed._budget_sig = model._budget_sig
        # add the dummy variable and the dummy constraint
        d = relaxed.addVar(name='dummy', vtype=gp.GRB.BINARY)
        relaxed.addConstr(d == 0, name='dummy')
        return relaxed

    @staticmethod
    def _get_relaxed_sol(model):
        if model.status == 9:
            return {}, {}, {}
        v = model.getAttr('x', model._v)
        y = model.getAttr('x', model._y)
        s = model.getAttr('x', model._s) if model._budget_sig > 0 else model._s.copy()
        return v, y, s

    def _add_cheap_cuts(self, model, cuts, v, y, s):
        cnt = 0
        not_opt = len(y) == 0
        for idx, cut in enumerate(cuts):
            if not_opt or (self._binding(cut, v, y, s)):
                cnt += 1
                rhs = cut['c'] + model._y.prod(cut['y'])
                if 's' in cut:
                    rhs += model._s.prod(cut['s'])
                model.addConstr(model._v[cut['od']] >= rhs, name='cheap{}'.format(idx))
        return cnt

    def _binding(self, cut, v, y, s):
        y_prod = np.sum([y[idx] * val for idx, val in cut['y'].items()])
        s_prod = np.sum([s[idx] * val for idx, val in cut['s'].items()]) if 's' in cut else 0
        rhs = cut['c'] + y_prod + s_prod
        return (v[cut['od']] > rhs - 1e-9) and (v[cut['od']] < rhs + 1e-9)

    def _construct_subproblems(self, od_pairs, projs, sig_costs, G, travel_time, M, regenerate):
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
        G_reverse = G.reverse()
        probs = {}
        for orig, des in tqdm(od_pairs):
            probs[(orig, des)] = self._subproblem(orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate)
        return probs

    def _subproblem(self, orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/continuous/sub_{}_{}.mps'.format(self.ins_name, orig, des)
        if file_existence(dir_name) and (not regenerate):
            model = gp.read(dir_name)
            model = self._set_sub_params(model)
            lamb, theta, gamma = self._get_sub_variables(model, proj_edges, unsig_cross, G)
            model = self._store_sub_variables(model, lamb, theta, gamma)
        else:
            # initialize the sub-problem
            model = gp.Model('subproblem_{}-{}'.format(orig, des))
            model = self._set_sub_params(model)
            # parameters
            relevant_nodes, relevant_edges = self._get_relevant_nodes_edges(G, G_reverse, orig, des, M)
            proj_edges = set(proj_edges)
            theta_edges = [(i, j) for i, j in relevant_edges if (i, j) in proj_edges]
            relevant_edges = set(relevant_edges)
            gamma_unsig = [(i, j, g, h) for (i, j, g, h) in unsig_cross if (i, j) in relevant_edges]
            relevant_edges = list(relevant_edges)
            # add variables
            lamb = model.addVars(relevant_nodes, name='lambda', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            theta = model.addVars(theta_edges, name='theta', lb=0, vtype=gp.GRB.CONTINUOUS)
            gamma = model.addVars(gamma_unsig, name='gamma', lb=0, vtype=gp.GRB.CONTINUOUS)
            # add constraints - x
            for i, j in relevant_edges:
                exp = - lamb[j] + lamb[i]
                if (i, j) in theta:
                    exp -= theta[i, j]
                if i in unsig_set:
                    exp -= gp.quicksum(gamma[i, j, g, h] for g, h in G.edges(i) if self._diff_edges(i, j, g, h) and (g, h) in theta)
                model.addConstr(exp <= travel_time[i, j], name='x')
            # add constraints - z
            model.addConstr(-lamb[des] + lamb[orig] <= M, name='f')
            # store decision variables
            model = self._store_sub_variables(model, lamb, theta, gamma)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_sub_params(model):
        model.Params.outputFlag = 0
        return model

    @staticmethod
    def _get_sub_variables(model, proj_edges, unsig_cross, G):
        lamb = {i: model.getVarByName("lambda[{}]".format(i)) for i in G.nodes}
        theta = {(i, j): model.getVarByName("theta[{},{}]".format(i, j)) for (i, j) in proj_edges}
        gamma = {(i, j, g, h): model.getVarByName("gamma[{},{},{},{}]".format(i, j, g, h)) for (i, j, g, h) in unsig_cross}
        return lamb, theta, gamma

    @staticmethod
    def _store_sub_variables(model, lamb, theta, gamma):
        model._lamb = lamb
        model._theta = theta
        model._gamma = gamma
        return model

    @staticmethod
    def _get_relevant_nodes_edges(G, G_reverse, orig, des, M):
        reachable_orig = single_source_dijkstra_path_length(G=G, source=orig, cutoff=M, weight='time')
        reachable_des = single_source_dijkstra_path_length(G=G_reverse, source=des, cutoff=M, weight='time')
        relevant_nodes = set([])
        for node, val in reachable_orig.items():
            if (node in reachable_des) and (reachable_des[node] + val <= M):
                relevant_nodes.add(node)
                for (node, tnode) in G.out_edges(node):
                    relevant_nodes.add(tnode)
                for (fnode, node) in G.in_edges(node):
                    relevant_nodes.add(fnode)
        relevant_edges = []
        for fnode, tnode in G.edges():
            if (fnode in relevant_nodes) and (tnode in relevant_nodes):
                relevant_edges.append((fnode, tnode))
        return relevant_nodes, relevant_edges

    def _gen_pareto_problems(self, duals, n_nodes, projs, sig_costs, G):
        proj_edges = flatten(projs)
        proj_edge_set = self._proj_edge_set(proj_edges)
        unsig_set = set(sig_costs.keys())
        unsig_cross = [(i, j, g, h) for s in unsig_set for (i, j) in list(G.out_edges(s))
                       for (g, h) in list(G.in_edges(s)) + list(G.out_edges(s))
                       if self._diff_edges(i, j, g, h) and ('{}_{}'.format(g, h) in proj_edge_set)]
        print('generating pareto problem ...')
        probs = {}
        for orig, des in tqdm(duals):
            probs[orig, des] = self._pareto_problem(duals[orig, des], n_nodes, proj_edges, unsig_cross)
        return probs

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
        # penalties = model.getAttr('x', model._p)
        y_val = model.getAttr('x', model._y)
        new_projects = [i for i in model._y if y_val[i] >= 1 - 1e-5]
        if model._budget_sig > 0:
            s_val = model.getAttr('x', model._s)
            new_signals = [i for i in model._s if s_val[i] >= 1 - 1e-5]
        else:
            new_signals = []
        return new_projects, new_signals


class BendersRegressionSolver(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, beta_1,
              insample_weight=1, reg_factor=1, loss_bound=1, manual_params={},
              quiet=False, time_limit=None, regenerate=False, pareto=False,
              relax4cut=False, weighted=False, mip_gap=None):
        """
        solve the optimization problem
        :param args: instance arguments
        :param budget_proj:
        :param budget_sig:
        :param time_limit: solution time limit
        :return: lists of connected pairs, new projects, and new signals
        """
        if not quiet:
            print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T, M = self._args2params(args)
        weights = args['weights'] if 'weights' in args and weighted else {}
        beta = self._gen_betas(beta_1, T, M)
        if not quiet:
            print('The instance has {} od-pairs, {} projects, {} candiate intersections, {} edges, and {} nodes'.format(
                len(od_pairs), len(projs), len(sig_costs), len(G.edges()), n_nodes))
            print('compiling ...')
        tick = time.time()
        re_solve = True
        if len(manual_params) > 0:
            T, M, beta = manual_params['T'], manual_params['M'], manual_params['beta']
        while re_solve:
            print('solve with loss bound: {:.4f}'.format(loss_bound / len(od_pairs)))
            master = self._construct_master_problem(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig,
                                                    T=T, beta=beta, weights=weights, proj_costs=proj_costs, sig_costs=sig_costs,
                                                    time_limit=time_limit, regenerate=regenerate, quiet=quiet, mip_gap=mip_gap,
                                                    reg_factor=1, loss_bound=loss_bound, insample_weight=insample_weight, args=args)
            if not quiet:
                master.update()
                print('Master problem has {} variables and {} constraints'.format(master.numVars, master.numConstrs))
            master._edge2proj = edge2proj
            master._T = T
            master._M = M
            master._y_hat, master._s_hat = self._find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs)
            master._sp = self._construct_subproblems(od_pairs, projs, sig_costs, G, travel_time, M, regenerate)
            if not quiet:
                master._sp[od_pairs[0]].update()
                print('A typical sub-problem problem has {} variables and {} constraints'.format(
                    master._sp[od_pairs[0]].numVars, master._sp[od_pairs[0]].numConstrs))
            master._pareto = pareto
            master._cnt = 0
            if not quiet:
                print('  elapsed: {:.2f} sec'.format(time.time() - tick))
                print('solving ...')
            tick = time.time()
            if relax4cut:
                if not quiet:
                    print('solving root relaxation for cheap cuts ...')
                cuts, v, y, s = self._relax4cut(master)
                cut_cnt = self._add_cheap_cuts(master, cuts, v, y, s)
                if not quiet:
                    print('{} cuts found, {} of them added'.format(len(cuts), cut_cnt))
            n_cheap_cuts = cut_cnt if relax4cut else 0
            master.optimize(benders_cut)
            t_sol = time.time() - tick
            if master.status == 3:
                loss_bound += len(od_pairs) * 0.1
            else:
                re_solve = False

        print('  obj val: {:.2f}'.format(master.objVal))
        print('  # of cuts added: {} = {} + {}'.format(n_cheap_cuts + master._cnt, n_cheap_cuts, master._cnt))
        print('  elapsed: {:.2f} sec'.format(t_sol))
        new_projects, new_signals = self._get_solution(master)
        return new_projects, new_signals, t_sol, loss_bound

    def _construct_master_problem(self, od_pairs, pop, budget_proj, budget_sig, proj_costs, sig_costs, beta, weights, T, time_limit, regenerate, quiet, mip_gap,
                                  insample_weight, reg_factor, loss_bound, args):
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
        dir_name = './prob/{}/models/continuous/master.mps'.format(self.ins_name)
        if file_existence(dir_name) and (not regenerate):
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            u, v, y, s = self._get_master_variables(model, od_pairs, list(proj_costs.keys()), list(sig_costs.keys()), budget_sig)
            # store variables
            model._u = u
            model._v = v
            model._y = y
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in list(sig_costs.keys())}
            model._budget_sig = budget_sig
            model = self._update_master_rhs(model, budget_proj, budget_sig)
        else:
            # get feature vector for out-of-sample prediction
            in_sample_feature = args['in_sample']
            out_of_sample_feature = args['out_of_sample']
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            # set parameters
            obj_weights = {(orig, des): pop[des] for orig, des in od_pairs}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            dimensions = list(range(len(out_of_sample_feature)))
            # add variables
            u = model.addVars(od_pairs, name='u', vtype=gp.GRB.CONTINUOUS)
            v = model.addVars(od_pairs, name='v', vtype=gp.GRB.CONTINUOUS)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            w = model.addVars(dimensions, name='w', vtype=gp.GRB.CONTINUOUS, lb=-reg_factor, ub=reg_factor)
            a = model.addVars(dimensions, name='a', vtype=gp.GRB.CONTINUOUS)
            p = model.addVars(od_pairs, name='p', vtype=gp.GRB.CONTINUOUS)
            if budget_sig > 0:
                s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
            # add budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            if budget_sig > 0:
                model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # add time constraints
            model.addConstrs((u[orig, des] >= v[orig, des] - T for orig, des in od_pairs), name='time_exceeds')
            # regularization
            model.addConstrs((a[i] >= w[i] for i in dimensions), name='reg_pos')
            model.addConstrs((a[i] >= -w[i] for i in dimensions), name='reg_neg')
            model.addConstr(a.sum() <= reg_factor, name='reg')
            # training loss
            model.addConstrs((p[orig, des] >= w.prod(in_sample_feature[orig, des]) - beta[1] * v[orig, des] - (beta[2] - beta[1]) * u[orig, des]
                              for orig, des in od_pairs), name='loss_pos')
            model.addConstrs((p[orig, des] >= beta[1] * v[orig, des] + (beta[2] - beta[1]) * u[orig, des] - w.prod(in_sample_feature[orig, des])
                              for orig, des in od_pairs), name='loss_neg')
            model.addConstr(p.sum() <= loss_bound, name='loss_bound')
            # set objective
            obj = insample_weight * (beta[1] * v.prod(obj_weights) + (beta[2] - beta[1]) * u.prod(obj_weights)) + w.prod(out_of_sample_feature)
            model.setObjective(obj, gp.GRB.MINIMIZE)
            # add variable dicts
            model._u = u
            model._v = v
            model._y = y
            model._w = w
            model._budget_sig = budget_sig
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in signals}
            # write the problem to local drive
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_master_params(model, time_limit, quiet, mip_gap):
        model.Params.outputFlag = 0 if quiet else 1
        if time_limit:
            model.Params.timeLimit = time_limit
        if mip_gap:
            model.Params.mipGap = mip_gap
        model.Params.lazyConstraints = 1
        # model.Params.presolve = 0
        return model

    @staticmethod
    def _get_master_variables(model, od_pairs, projects, signals, budget_sig):
        u = gp.tupledict([((orig, des), model.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        v = gp.tupledict([((orig, des), model.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        y = gp.tupledict([(i, model.getVarByName("y[{}]".format(i))) for i in projects])
        if budget_sig > 0:
            s = gp.tupledict([(i, model.getVarByName("s[{}]".format(i))) for i in signals])
        else:
            s = []
        return u, v, y, s

    @staticmethod
    def _update_master_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

    @staticmethod
    def _find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs):
        """
        find the raltiave inner points (for the pareto cut problem)
        :param budget_proj:
        :param budget_sig:
        :param proj_costs:
        :param sig_costs:
        :return: dict for y, dict for s
        """
        n_proj = len(proj_costs)
        n_sig = len(sig_costs)
        y_hat = {idx: np.min([1, budget_proj / (2 * n_proj * val)]) for idx, val in proj_costs.items()}
        if budget_sig > 0:
            s_hat = {idx: np.min([1, budget_sig / (2 * n_sig * val)]) for idx, val in sig_costs.items()}
        else:
            s_hat = {idx: 0 for idx, val in sig_costs.items()}
        return y_hat, s_hat

    def _relax4cut(self, model):
        # initialize the relaxed problem
        relaxed = self._gen_relaxed_problem(model)
        relaxed.Params.outputFlag = 0
        relaxed.optimize(benders_cut_root_relax)
        v, y, s = self._get_relaxed_sol(relaxed)
        return relaxed._coeffs, v, y, s

    @staticmethod
    def _gen_relaxed_problem(model):
        model.update()
        relaxed = model.relax()
        relaxed._u = gp.tupledict([((orig, des), relaxed.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in model._u])
        relaxed._v = gp.tupledict([((orig, des), relaxed.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in model._v])
        relaxed._y = gp.tupledict([(i, relaxed.getVarByName("y[{}]".format(i))) for i in model._y])
        if model._budget_sig > 0:
            relaxed._s = gp.tupledict([(i, relaxed.getVarByName("s[{}]".format(i))) for i in model._s])
        else:
            relaxed._s = model._s.copy()
        # store parameters
        relaxed._M = model._M
        relaxed._edge2proj = model._edge2proj
        relaxed._cnt = 0
        relaxed._sp = model._sp
        relaxed._coeffs = []
        relaxed._y_hat = model._y_hat
        relaxed._s_hat = model._s_hat
        relaxed._budget_sig = model._budget_sig
        # add the dummy variable and the dummy constraint
        d = relaxed.addVar(name='dummy', vtype=gp.GRB.BINARY)
        relaxed.addConstr(d == 0, name='dummy')
        return relaxed

    @staticmethod
    def _get_relaxed_sol(model):
        if model.status == 9:
            return {}, {}, {}
        v = model.getAttr('x', model._v)
        y = model.getAttr('x', model._y)
        s = model.getAttr('x', model._s) if model._budget_sig > 0 else model._s.copy()
        return v, y, s

    def _add_cheap_cuts(self, model, cuts, v, y, s):
        cnt = 0
        not_opt = len(y) == 0
        for idx, cut in enumerate(cuts):
            if not_opt or (self._binding(cut, v, y, s)):
                cnt += 1
                rhs = cut['c'] + model._y.prod(cut['y'])
                if 's' in cut:
                    rhs += model._s.prod(cut['s'])
                model.addConstr(model._v[cut['od']] >= rhs, name='cheap{}'.format(idx))
        return cnt

    def _binding(self, cut, v, y, s):
        y_prod = np.sum([y[idx] * val for idx, val in cut['y'].items()])
        s_prod = np.sum([s[idx] * val for idx, val in cut['s'].items()]) if 's' in cut else 0
        rhs = cut['c'] + y_prod + s_prod
        return (v[cut['od']] > rhs - 1e-9) and (v[cut['od']] < rhs + 1e-9)

    def _construct_subproblems(self, od_pairs, projs, sig_costs, G, travel_time, M, regenerate):
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
        G_reverse = G.reverse()
        probs = {}
        for orig, des in tqdm(od_pairs):
            probs[(orig, des)] = self._subproblem(orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate)
        return probs

    def _subproblem(self, orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/continuous/sub_{}_{}.mps'.format(self.ins_name, orig, des)
        if file_existence(dir_name) and (not regenerate):
            model = gp.read(dir_name)
            model = self._set_sub_params(model)
            lamb, theta, gamma = self._get_sub_variables(model, proj_edges, unsig_cross, G)
            model = self._store_sub_variables(model, lamb, theta, gamma)
        else:
            # initialize the sub-problem
            model = gp.Model('subproblem_{}-{}'.format(orig, des))
            model = self._set_sub_params(model)
            # parameters
            relevant_nodes, relevant_edges = self._get_relevant_nodes_edges(G, G_reverse, orig, des, M)
            proj_edges = set(proj_edges)
            theta_edges = [(i, j) for i, j in relevant_edges if (i, j) in proj_edges]
            relevant_edges = set(relevant_edges)
            gamma_unsig = [(i, j, g, h) for (i, j, g, h) in unsig_cross if (i, j) in relevant_edges]
            relevant_edges = list(relevant_edges)
            # add variables
            lamb = model.addVars(relevant_nodes, name='lambda', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            theta = model.addVars(theta_edges, name='theta', lb=0, vtype=gp.GRB.CONTINUOUS)
            gamma = model.addVars(gamma_unsig, name='gamma', lb=0, vtype=gp.GRB.CONTINUOUS)
            # add constraints - x
            for i, j in relevant_edges:
                exp = - lamb[j] + lamb[i]
                if (i, j) in theta:
                    exp -= theta[i, j]
                if i in unsig_set:
                    exp -= gp.quicksum(gamma[i, j, g, h] for g, h in G.edges(i) if self._diff_edges(i, j, g, h) and (g, h) in theta)
                model.addConstr(exp <= travel_time[i, j], name='x')
            # add constraints - z
            model.addConstr(-lamb[des] + lamb[orig] <= M, name='f')
            # store decision variables
            model = self._store_sub_variables(model, lamb, theta, gamma)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_sub_params(model):
        model.Params.outputFlag = 0
        return model

    @staticmethod
    def _get_sub_variables(model, proj_edges, unsig_cross, G):
        lamb = {i: model.getVarByName("lambda[{}]".format(i)) for i in G.nodes}
        theta = {(i, j): model.getVarByName("theta[{},{}]".format(i, j)) for (i, j) in proj_edges}
        gamma = {(i, j, g, h): model.getVarByName("gamma[{},{},{},{}]".format(i, j, g, h)) for (i, j, g, h) in unsig_cross}
        return lamb, theta, gamma

    @staticmethod
    def _store_sub_variables(model, lamb, theta, gamma):
        model._lamb = lamb
        model._theta = theta
        model._gamma = gamma
        return model

    @staticmethod
    def _get_relevant_nodes_edges(G, G_reverse, orig, des, M):
        reachable_orig = single_source_dijkstra_path_length(G=G, source=orig, cutoff=M, weight='time')
        reachable_des = single_source_dijkstra_path_length(G=G_reverse, source=des, cutoff=M, weight='time')
        relevant_nodes = set([])
        for node, val in reachable_orig.items():
            if (node in reachable_des) and (reachable_des[node] + val <= M):
                relevant_nodes.add(node)
                for (node, tnode) in G.out_edges(node):
                    relevant_nodes.add(tnode)
                for (fnode, node) in G.in_edges(node):
                    relevant_nodes.add(fnode)
        relevant_edges = []
        for fnode, tnode in G.edges():
            if (fnode in relevant_nodes) and (tnode in relevant_nodes):
                relevant_edges.append((fnode, tnode))
        return relevant_nodes, relevant_edges

    def _gen_pareto_problems(self, duals, n_nodes, projs, sig_costs, G):
        proj_edges = flatten(projs)
        proj_edge_set = self._proj_edge_set(proj_edges)
        unsig_set = set(sig_costs.keys())
        unsig_cross = [(i, j, g, h) for s in unsig_set for (i, j) in list(G.out_edges(s))
                       for (g, h) in list(G.in_edges(s)) + list(G.out_edges(s))
                       if self._diff_edges(i, j, g, h) and ('{}_{}'.format(g, h) in proj_edge_set)]
        print('generating pareto problem ...')
        probs = {}
        for orig, des in tqdm(duals):
            probs[orig, des] = self._pareto_problem(duals[orig, des], n_nodes, proj_edges, unsig_cross)
        return probs

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
        # penalties = model.getAttr('x', model._p)
        y_val = model.getAttr('x', model._y)
        new_projects = [i for i in model._y if y_val[i] >= 1 - 1e-5]
        if model._budget_sig > 0:
            s_val = model.getAttr('x', model._s)
            new_signals = [i for i in model._s if s_val[i] >= 1 - 1e-5]
        else:
            new_signals = []
        return new_projects, new_signals


class BendersRegressionSolverVariant(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, beta_1,
              insample_weight=1, reg_factor=1, loss_bound=1, manual_params={},
              quiet=False, time_limit=None, regenerate=False, pareto=False,
              relax4cut=False, weighted=False, mip_gap=None):
        """
        solve the optimization problem
        :param args: instance arguments
        :param budget_proj:
        :param budget_sig:
        :param time_limit: solution time limit
        :return: lists of connected pairs, new projects, and new signals
        """
        if not quiet:
            print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T, M = self._args2params(args)
        weights = args['weights'] if 'weights' in args and weighted else {}
        beta = self._gen_betas(beta_1, T, M)
        if not quiet:
            print('The instance has {} od-pairs, {} projects, {} candiate intersections, {} edges, and {} nodes'.format(
                len(od_pairs), len(projs), len(sig_costs), len(G.edges()), n_nodes))
            print('compiling ...')
        tick = time.time()
        re_solve = True
        if len(manual_params) > 0:
            T, M, beta = manual_params['T'], manual_params['M'], manual_params['beta']
        while re_solve:
            print('solve with loss bound: {:.4f}'.format(loss_bound / len(od_pairs)))
            master = self._construct_master_problem(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig,
                                                    T=T, M=M, beta=beta, weights=weights, proj_costs=proj_costs, sig_costs=sig_costs,
                                                    time_limit=time_limit, regenerate=regenerate, quiet=quiet, mip_gap=mip_gap,
                                                    reg_factor=1, loss_bound=loss_bound, insample_weight=insample_weight, args=args)
            if not quiet:
                master.update()
                print('Master problem has {} variables and {} constraints'.format(master.numVars, master.numConstrs))
            master._edge2proj = edge2proj
            master._T = T
            master._M = M
            master._y_hat, master._s_hat = self._find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs)
            master._sp = self._construct_subproblems(od_pairs, projs, sig_costs, G, travel_time, M, regenerate)
            if not quiet:
                master._sp[od_pairs[0]].update()
                print('A typical sub-problem problem has {} variables and {} constraints'.format(
                    master._sp[od_pairs[0]].numVars, master._sp[od_pairs[0]].numConstrs))
            master._pareto = pareto
            master._cnt = 0
            if not quiet:
                print('  elapsed: {:.2f} sec'.format(time.time() - tick))
                print('solving ...')
            tick = time.time()
            if relax4cut:
                if not quiet:
                    print('solving root relaxation for cheap cuts ...')
                cuts, v, y, s = self._relax4cut(master)
                cut_cnt = self._add_cheap_cuts(master, cuts, v, y, s)
                if not quiet:
                    print('{} cuts found, {} of them added'.format(len(cuts), cut_cnt))
            n_cheap_cuts = cut_cnt if relax4cut else 0
            master.optimize(benders_cut)
            t_sol = time.time() - tick
            if master.status == 3:
                loss_bound += len(od_pairs) * 0.1
            else:
                re_solve = False

        print('  obj val: {:.2f}'.format(master.objVal))
        print('  # of cuts added: {} = {} + {}'.format(n_cheap_cuts + master._cnt, n_cheap_cuts, master._cnt))
        print('  elapsed: {:.2f} sec'.format(t_sol))
        new_projects, new_signals = self._get_solution(master)
        return new_projects, new_signals, t_sol, loss_bound

    def _construct_master_problem(self, od_pairs, pop, budget_proj, budget_sig, proj_costs, sig_costs, beta, weights,
                                  T, M, time_limit, regenerate, quiet, mip_gap, insample_weight, reg_factor, loss_bound,
                                  args):
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
        dir_name = './prob/{}/models/continuous/master.mps'.format(self.ins_name)
        if file_existence(dir_name) and (not regenerate):
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            u, v, y, s = self._get_master_variables(model, od_pairs, list(proj_costs.keys()), list(sig_costs.keys()), budget_sig)
            # store variables
            model._u = u
            model._v = v
            model._y = y
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in list(sig_costs.keys())}
            model._budget_sig = budget_sig
            model = self._update_master_rhs(model, budget_proj, budget_sig)
        else:
            convex = beta[1] >= beta[2]
            # get feature vector for out-of-sample prediction
            in_sample_feature = args['in_sample']
            out_of_sample_feature = args['out_of_sample']
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            # set parameters
            obj_weights = {(orig, des): pop[des] for orig, des in od_pairs}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            dimensions = list(range(len(out_of_sample_feature)))
            # add variables
            v = model.addVars(od_pairs, name='v', vtype=gp.GRB.CONTINUOUS)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            w = model.addVars(dimensions, name='w', vtype=gp.GRB.CONTINUOUS, lb=-reg_factor, ub=reg_factor)
            a = model.addVars(dimensions, name='a', vtype=gp.GRB.CONTINUOUS)
            p = model.addVars(od_pairs, name='p', vtype=gp.GRB.CONTINUOUS)
            if convex:
                x = model.addVars(od_pairs, name='x', ub=1, lb=0, vtype=gp.GRB.CONTINUOUS)
                z = model.addVars(od_pairs, 2, name='z', vtype=gp.GRB.CONTINUOUS)
            else:
                u = model.addVars(od_pairs, name='u', vtype=gp.GRB.CONTINUOUS)
            if budget_sig > 0:
                s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
            # add budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            if budget_sig > 0:
                model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # add time constraints
            if convex:
                model.addConstrs((z[orig, des, 0] <= T * x[orig, des] for orig, des in od_pairs), name='interval_0_leq')
                model.addConstrs((z[orig, des, 1] <= M * (1 - x[orig, des])
                                  for orig, des in od_pairs), name='interval_1_leq')
                model.addConstrs((z[orig, des, 1] >= T * (1 - x[orig, des])
                                  for orig, des in od_pairs), name='interval_1_geq')
                model.addConstrs((z[orig, des, 0] + z[orig, des, 1] == v[orig, des]
                                  for orig, des in od_pairs), name='sum_w')
                # training loss
                intercept = {0: beta[0], 1: beta[0] - beta[1] * T + beta[2] * T}
                slope = {0: beta[1], 1: beta[2]}
                model.addConstrs((p[orig, des] >= w.prod(in_sample_feature[orig, des])
                                  - gp.quicksum(- intercept[idx] * x[orig, des] + slope[idx] * z[orig, des, idx] for idx in range(2))
                                  for orig, des in od_pairs), name='loss_pos')
                model.addConstrs((p[orig, des] >= gp.quicksum(- intercept[idx] * x[orig, des] + slope[idx] * z[orig, des, idx] for idx in range(2))
                                  - w.prod(in_sample_feature[orig, des])
                                  for orig, des in od_pairs), name='loss_neg')
                model.addConstr(p.sum() <= loss_bound, name='loss_bound')
                # set obj
                obj = insample_weight * gp.quicksum((-intercept[idx] * x[orig, des] + slope[idx] * z[orig, des, idx])
                                                    * obj_weights[orig, des] for (orig, des) in od_pairs for idx in range(2)) \
                      + w.prod(out_of_sample_feature)
            else:
                model.addConstrs((u[orig, des] >= v[orig, des] - T for orig, des in od_pairs), name='time_exceeds')
                # training loss
                model.addConstrs((p[orig, des] >= w.prod(in_sample_feature[orig, des]) - beta[1] * v[orig, des] - (beta[2] - beta[1]) * u[orig, des] for orig, des in od_pairs), name='loss_pos')
                model.addConstrs((p[orig, des] >= beta[1] * v[orig, des] + (beta[2] - beta[1]) * u[orig, des] - w.prod(in_sample_feature[orig, des]) for orig, des in od_pairs), name='loss_neg')
                model.addConstr(p.sum() <= loss_bound, name='loss_bound')
                # set objective
                obj = insample_weight * (beta[1] * v.prod(obj_weights) + (beta[2] - beta[1]) * u.prod(obj_weights)) + w.prod(out_of_sample_feature)

            # regularization
            model.addConstrs((a[i] >= w[i] for i in dimensions), name='reg_pos')
            model.addConstrs((a[i] >= -w[i] for i in dimensions), name='reg_neg')
            model.addConstr(a.sum() <= reg_factor, name='reg')
            # set obj
            model.setObjective(obj, gp.GRB.MINIMIZE)
            # add variable dicts
            # model._u = u
            model._v = v
            model._y = y
            model._w = w
            model._budget_sig = budget_sig
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in signals}
            # write the problem to local drive
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_master_params(model, time_limit, quiet, mip_gap):
        model.Params.outputFlag = 0 if quiet else 1
        if time_limit:
            model.Params.timeLimit = time_limit
        if mip_gap:
            model.Params.mipGap = mip_gap
        model.Params.lazyConstraints = 1
        # model.Params.presolve = 0
        return model

    @staticmethod
    def _get_master_variables(model, od_pairs, projects, signals, budget_sig):
        u = gp.tupledict([((orig, des), model.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        v = gp.tupledict([((orig, des), model.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        y = gp.tupledict([(i, model.getVarByName("y[{}]".format(i))) for i in projects])
        if budget_sig > 0:
            s = gp.tupledict([(i, model.getVarByName("s[{}]".format(i))) for i in signals])
        else:
            s = []
        return u, v, y, s

    @staticmethod
    def _update_master_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

    @staticmethod
    def _find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs):
        """
        find the raltiave inner points (for the pareto cut problem)
        :param budget_proj:
        :param budget_sig:
        :param proj_costs:
        :param sig_costs:
        :return: dict for y, dict for s
        """
        n_proj = len(proj_costs)
        n_sig = len(sig_costs)
        y_hat = {idx: np.min([1, budget_proj / (2 * n_proj * val)]) for idx, val in proj_costs.items()}
        if budget_sig > 0:
            s_hat = {idx: np.min([1, budget_sig / (2 * n_sig * val)]) for idx, val in sig_costs.items()}
        else:
            s_hat = {idx: 0 for idx, val in sig_costs.items()}
        return y_hat, s_hat

    def _relax4cut(self, model):
        # initialize the relaxed problem
        relaxed = self._gen_relaxed_problem(model)
        relaxed.Params.outputFlag = 0
        relaxed.optimize(benders_cut_root_relax)
        v, y, s = self._get_relaxed_sol(relaxed)
        return relaxed._coeffs, v, y, s

    @staticmethod
    def _gen_relaxed_problem(model):
        model.update()
        relaxed = model.relax()
        # relaxed._u = gp.tupledict([((orig, des), relaxed.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in model._u])
        relaxed._v = gp.tupledict([((orig, des), relaxed.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in model._v])
        relaxed._y = gp.tupledict([(i, relaxed.getVarByName("y[{}]".format(i))) for i in model._y])
        if model._budget_sig > 0:
            relaxed._s = gp.tupledict([(i, relaxed.getVarByName("s[{}]".format(i))) for i in model._s])
        else:
            relaxed._s = model._s.copy()
        # store parameters
        relaxed._M = model._M
        relaxed._edge2proj = model._edge2proj
        relaxed._cnt = 0
        relaxed._sp = model._sp
        relaxed._coeffs = []
        relaxed._y_hat = model._y_hat
        relaxed._s_hat = model._s_hat
        relaxed._budget_sig = model._budget_sig
        # add the dummy variable and the dummy constraint
        d = relaxed.addVar(name='dummy', vtype=gp.GRB.BINARY)
        relaxed.addConstr(d == 0, name='dummy')
        return relaxed

    @staticmethod
    def _get_relaxed_sol(model):
        if model.status == 9 or model.status == 3:
            return {}, {}, {}
        v = model.getAttr('x', model._v)
        y = model.getAttr('x', model._y)
        s = model.getAttr('x', model._s) if model._budget_sig > 0 else model._s.copy()
        return v, y, s

    def _add_cheap_cuts(self, model, cuts, v, y, s):
        cnt = 0
        not_opt = len(y) == 0
        for idx, cut in enumerate(cuts):
            if not_opt or (self._binding(cut, v, y, s)):
                cnt += 1
                rhs = cut['c'] + model._y.prod(cut['y'])
                if 's' in cut:
                    rhs += model._s.prod(cut['s'])
                model.addConstr(model._v[cut['od']] >= rhs, name='cheap{}'.format(idx))
        return cnt

    def _binding(self, cut, v, y, s):
        y_prod = np.sum([y[idx] * val for idx, val in cut['y'].items()])
        s_prod = np.sum([s[idx] * val for idx, val in cut['s'].items()]) if 's' in cut else 0
        rhs = cut['c'] + y_prod + s_prod
        return (v[cut['od']] > rhs - 1e-9) and (v[cut['od']] < rhs + 1e-9)

    def _construct_subproblems(self, od_pairs, projs, sig_costs, G, travel_time, M, regenerate):
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
        G_reverse = G.reverse()
        probs = {}
        for orig, des in tqdm(od_pairs):
            probs[(orig, des)] = self._subproblem(orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate)
        return probs

    def _subproblem(self, orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/continuous/sub_{}_{}.mps'.format(self.ins_name, orig, des)
        if file_existence(dir_name) and (not regenerate):
            model = gp.read(dir_name)
            model = self._set_sub_params(model)
            lamb, theta, gamma = self._get_sub_variables(model, proj_edges, unsig_cross, G)
            model = self._store_sub_variables(model, lamb, theta, gamma)
        else:
            # initialize the sub-problem
            model = gp.Model('subproblem_{}-{}'.format(orig, des))
            model = self._set_sub_params(model)
            # parameters
            relevant_nodes, relevant_edges = self._get_relevant_nodes_edges(G, G_reverse, orig, des, M)
            proj_edges = set(proj_edges)
            theta_edges = [(i, j) for i, j in relevant_edges if (i, j) in proj_edges]
            relevant_edges = set(relevant_edges)
            gamma_unsig = [(i, j, g, h) for (i, j, g, h) in unsig_cross if (i, j) in relevant_edges]
            relevant_edges = list(relevant_edges)
            # add variables
            lamb = model.addVars(relevant_nodes, name='lambda', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            theta = model.addVars(theta_edges, name='theta', lb=0, vtype=gp.GRB.CONTINUOUS)
            gamma = model.addVars(gamma_unsig, name='gamma', lb=0, vtype=gp.GRB.CONTINUOUS)
            # add constraints - x
            for i, j in relevant_edges:
                exp = - lamb[j] + lamb[i]
                if (i, j) in theta:
                    exp -= theta[i, j]
                if i in unsig_set:
                    exp -= gp.quicksum(gamma[i, j, g, h] for g, h in G.edges(i) if self._diff_edges(i, j, g, h) and (g, h) in theta)
                model.addConstr(exp <= travel_time[i, j], name='x')
            # add constraints - z
            model.addConstr(-lamb[des] + lamb[orig] <= M, name='f')
            # store decision variables
            model = self._store_sub_variables(model, lamb, theta, gamma)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_sub_params(model):
        model.Params.outputFlag = 0
        return model

    @staticmethod
    def _get_sub_variables(model, proj_edges, unsig_cross, G):
        lamb = {i: model.getVarByName("lambda[{}]".format(i)) for i in G.nodes}
        theta = {(i, j): model.getVarByName("theta[{},{}]".format(i, j)) for (i, j) in proj_edges}
        gamma = {(i, j, g, h): model.getVarByName("gamma[{},{},{},{}]".format(i, j, g, h)) for (i, j, g, h) in unsig_cross}
        return lamb, theta, gamma

    @staticmethod
    def _store_sub_variables(model, lamb, theta, gamma):
        model._lamb = lamb
        model._theta = theta
        model._gamma = gamma
        return model

    @staticmethod
    def _get_relevant_nodes_edges(G, G_reverse, orig, des, M):
        reachable_orig = single_source_dijkstra_path_length(G=G, source=orig, cutoff=M, weight='time')
        reachable_des = single_source_dijkstra_path_length(G=G_reverse, source=des, cutoff=M, weight='time')
        relevant_nodes = set([])
        for node, val in reachable_orig.items():
            if (node in reachable_des) and (reachable_des[node] + val <= M):
                relevant_nodes.add(node)
                for (node, tnode) in G.out_edges(node):
                    relevant_nodes.add(tnode)
                for (fnode, node) in G.in_edges(node):
                    relevant_nodes.add(fnode)
        relevant_edges = []
        for fnode, tnode in G.edges():
            if (fnode in relevant_nodes) and (tnode in relevant_nodes):
                relevant_edges.append((fnode, tnode))
        return relevant_nodes, relevant_edges

    def _gen_pareto_problems(self, duals, n_nodes, projs, sig_costs, G):
        proj_edges = flatten(projs)
        proj_edge_set = self._proj_edge_set(proj_edges)
        unsig_set = set(sig_costs.keys())
        unsig_cross = [(i, j, g, h) for s in unsig_set for (i, j) in list(G.out_edges(s))
                       for (g, h) in list(G.in_edges(s)) + list(G.out_edges(s))
                       if self._diff_edges(i, j, g, h) and ('{}_{}'.format(g, h) in proj_edge_set)]
        print('generating pareto problem ...')
        probs = {}
        for orig, des in tqdm(duals):
            probs[orig, des] = self._pareto_problem(duals[orig, des], n_nodes, proj_edges, unsig_cross)
        return probs

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
        # penalties = model.getAttr('x', model._p)
        y_val = model.getAttr('x', model._y)
        new_projects = [i for i in model._y if y_val[i] >= 1 - 1e-5]
        if model._budget_sig > 0:
            s_val = model.getAttr('x', model._s)
            new_signals = [i for i in model._s if s_val[i] >= 1 - 1e-5]
        else:
            new_signals = []
        return new_projects, new_signals


class BendersBaggingSolver(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, beta_1,
              knn_weight=1, reg_factor=1, loss_bound=1,
              quiet=False, time_limit=None, regenerate=False, pareto=False,
              relax4cut=False, weighted=False, mip_gap=None):
        """
        solve the optimization problem
        :param args: instance arguments
        :param budget_proj:
        :param budget_sig:
        :param time_limit: solution time limit
        :return: lists of connected pairs, new projects, and new signals
        """
        if not quiet:
            print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T, M = self._args2params(args)
        weights = args['weights'] if 'weights' in args and weighted else {}
        beta = self._gen_betas(beta_1, T, M)
        if not quiet:
            print('The instance has {} od-pairs, {} projects, {} candiate intersections, {} edges, and {} nodes'.format(
                len(od_pairs), len(projs), len(sig_costs), len(G.edges()), n_nodes))
            print('compiling ...')
        tick = time.time()
        master = self._construct_master_problem(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig,
                                                T=T, beta=beta, weights=weights, proj_costs=proj_costs, sig_costs=sig_costs,
                                                time_limit=time_limit, regenerate=regenerate, quiet=quiet, mip_gap=mip_gap,
                                                reg_factor=1, loss_bound=loss_bound, knn_weight=knn_weight, args=args)
        if not quiet:
            master.update()
            print('Master problem has {} variables and {} constraints'.format(master.numVars, master.numConstrs))
        master._edge2proj = edge2proj
        master._T = T
        master._M = M
        master._y_hat, master._s_hat = self._find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs)
        master._sp = self._construct_subproblems(od_pairs, projs, sig_costs, G, travel_time, M, regenerate)
        if not quiet:
            master._sp[od_pairs[0]].update()
            print('A typical sub-problem problem has {} variables and {} constraints'.format(
                master._sp[od_pairs[0]].numVars, master._sp[od_pairs[0]].numConstrs))
        master._pareto = pareto
        master._cnt = 0
        if not quiet:
            print('  elapsed: {:.2f} sec'.format(time.time() - tick))
            print('solving ...')
        tick = time.time()
        if relax4cut:
            if not quiet:
                print('solving root relaxation for cheap cuts ...')
            cuts, v, y, s = self._relax4cut(master)
            cut_cnt = self._add_cheap_cuts(master, cuts, v, y, s)
            if not quiet:
                print('{} cuts found, {} of them added'.format(len(cuts), cut_cnt))
        n_cheap_cuts = cut_cnt if relax4cut else 0
        master.optimize(benders_cut)
        t_sol = time.time() - tick
        print('  obj val: {:.2f}'.format(master.objVal))
        print('  # of cuts added: {} = {} + {}'.format(n_cheap_cuts + master._cnt, n_cheap_cuts, master._cnt))
        print('  elapsed: {:.2f} sec'.format(t_sol))
        new_projects, new_signals = self._get_solution(master)
        return new_projects, new_signals, t_sol

    def _construct_master_problem(self, od_pairs, pop, budget_proj, budget_sig, proj_costs, sig_costs, beta, weights, T, time_limit, regenerate, quiet, mip_gap,
                                  knn_weight, reg_factor, loss_bound, args):
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
        dir_name = './prob/{}/models/continuous/master.mps'.format(self.ins_name)
        if file_existence(dir_name) and (not regenerate):
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            u, v, y, s = self._get_master_variables(model, od_pairs, list(proj_costs.keys()), list(sig_costs.keys()), budget_sig)
            # store variables
            model._u = u
            model._v = v
            model._y = y
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in list(sig_costs.keys())}
            model._budget_sig = budget_sig
            model = self._update_master_rhs(model, budget_proj, budget_sig)
        else:
            # get feature vector for out-of-sample prediction
            in_sample_feature = args['in_sample']
            out_of_sample_feature = args['out_of_sample']
            obj_weights = {(orig, des): weights[orig, des] * knn_weight + (1 - knn_weight) * pop[des] for orig, des in weights}
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            # set parameters
            obj_weights = weights if len(weights) > 0 else {(orig, des): pop[des] for orig, des in od_pairs}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            dimensions = list(range(len(out_of_sample_feature)))
            # add variables
            u = model.addVars(od_pairs, name='u', vtype=gp.GRB.CONTINUOUS)
            v = model.addVars(od_pairs, name='v', vtype=gp.GRB.CONTINUOUS)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            w = model.addVars(dimensions, name='w', vtype=gp.GRB.CONTINUOUS, lb=-reg_factor, ub=reg_factor)
            a = model.addVars(dimensions, name='a', vtype=gp.GRB.CONTINUOUS)
            p = model.addVars(od_pairs, name='p', vtype=gp.GRB.CONTINUOUS)
            if budget_sig > 0:
                s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
            # add budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            if budget_sig > 0:
                model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # add time constraints
            model.addConstrs((u[orig, des] >= v[orig, des] - T for orig, des in od_pairs), name='time_exceeds')
            # regularization
            model.addConstrs((a[i] >= w[i] for i in dimensions), name='reg_pos')
            model.addConstrs((a[i] >= -w[i] for i in dimensions), name='reg_neg')
            model.addConstr(a.sum() <= reg_factor, name='reg')
            # training loss
            model.addConstrs((p[orig, des] >= w.prod(in_sample_feature[orig, des]) - beta[1] * v[orig, des] - (beta[2] - beta[1]) * u[orig, des]
                              for orig, des in od_pairs), name='loss_pos')
            model.addConstrs((p[orig, des] >= beta[1] * v[orig, des] + (beta[2] - beta[1]) * u[orig, des] - w.prod(in_sample_feature[orig, des])
                              for orig, des in od_pairs), name='loss_neg')
            model.addConstr(p.sum() <= loss_bound, name='loss_bound')
            # set objective
            obj = beta[1] * v.prod(obj_weights) + (beta[2] - beta[1]) * u.prod(obj_weights) + (1 - knn_weight) * w.prod(out_of_sample_feature)
            model.setObjective(obj, gp.GRB.MINIMIZE)
            # add variable dicts
            model._u = u
            model._v = v
            model._y = y
            model._w = w
            model._budget_sig = budget_sig
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in signals}
            # write the problem to local drive
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_master_params(model, time_limit, quiet, mip_gap):
        model.Params.outputFlag = 0 if quiet else 1
        if time_limit:
            model.Params.timeLimit = time_limit
        if mip_gap:
            model.Params.mipGap = mip_gap
        model.Params.lazyConstraints = 1
        # model.Params.presolve = 0
        return model

    @staticmethod
    def _get_master_variables(model, od_pairs, projects, signals, budget_sig):
        u = gp.tupledict([((orig, des), model.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        v = gp.tupledict([((orig, des), model.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        y = gp.tupledict([(i, model.getVarByName("y[{}]".format(i))) for i in projects])
        if budget_sig > 0:
            s = gp.tupledict([(i, model.getVarByName("s[{}]".format(i))) for i in signals])
        else:
            s = []
        return u, v, y, s

    @staticmethod
    def _update_master_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

    @staticmethod
    def _find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs):
        """
        find the raltiave inner points (for the pareto cut problem)
        :param budget_proj:
        :param budget_sig:
        :param proj_costs:
        :param sig_costs:
        :return: dict for y, dict for s
        """
        n_proj = len(proj_costs)
        n_sig = len(sig_costs)
        y_hat = {idx: np.min([1, budget_proj / (2 * n_proj * val)]) for idx, val in proj_costs.items()}
        if budget_sig > 0:
            s_hat = {idx: np.min([1, budget_sig / (2 * n_sig * val)]) for idx, val in sig_costs.items()}
        else:
            s_hat = {idx: 0 for idx, val in sig_costs.items()}
        return y_hat, s_hat

    def _relax4cut(self, model):
        # initialize the relaxed problem
        relaxed = self._gen_relaxed_problem(model)
        relaxed.Params.outputFlag = 0
        relaxed.optimize(benders_cut_root_relax)
        v, y, s = self._get_relaxed_sol(relaxed)
        return relaxed._coeffs, v, y, s

    @staticmethod
    def _gen_relaxed_problem(model):
        model.update()
        relaxed = model.relax()
        relaxed._u = gp.tupledict([((orig, des), relaxed.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in model._u])
        relaxed._v = gp.tupledict([((orig, des), relaxed.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in model._v])
        relaxed._y = gp.tupledict([(i, relaxed.getVarByName("y[{}]".format(i))) for i in model._y])
        if model._budget_sig > 0:
            relaxed._s = gp.tupledict([(i, relaxed.getVarByName("s[{}]".format(i))) for i in model._s])
        else:
            relaxed._s = model._s.copy()
        # store parameters
        relaxed._M = model._M
        relaxed._edge2proj = model._edge2proj
        relaxed._cnt = 0
        relaxed._sp = model._sp
        relaxed._coeffs = []
        relaxed._y_hat = model._y_hat
        relaxed._s_hat = model._s_hat
        relaxed._budget_sig = model._budget_sig
        # add the dummy variable and the dummy constraint
        d = relaxed.addVar(name='dummy', vtype=gp.GRB.BINARY)
        relaxed.addConstr(d == 0, name='dummy')
        return relaxed

    @staticmethod
    def _get_relaxed_sol(model):
        if model.status == 9:
            return {}, {}, {}
        v = model.getAttr('x', model._v)
        y = model.getAttr('x', model._y)
        s = model.getAttr('x', model._s) if model._budget_sig > 0 else model._s.copy()
        return v, y, s

    def _add_cheap_cuts(self, model, cuts, v, y, s):
        cnt = 0
        not_opt = len(y) == 0
        for idx, cut in enumerate(cuts):
            if not_opt or (self._binding(cut, v, y, s)):
                cnt += 1
                rhs = cut['c'] + model._y.prod(cut['y'])
                if 's' in cut:
                    rhs += model._s.prod(cut['s'])
                model.addConstr(model._v[cut['od']] >= rhs, name='cheap{}'.format(idx))
        return cnt

    def _binding(self, cut, v, y, s):
        y_prod = np.sum([y[idx] * val for idx, val in cut['y'].items()])
        s_prod = np.sum([s[idx] * val for idx, val in cut['s'].items()]) if 's' in cut else 0
        rhs = cut['c'] + y_prod + s_prod
        return (v[cut['od']] > rhs - 1e-9) and (v[cut['od']] < rhs + 1e-9)

    def _construct_subproblems(self, od_pairs, projs, sig_costs, G, travel_time, M, regenerate):
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
        G_reverse = G.reverse()
        probs = {}
        for orig, des in tqdm(od_pairs):
            probs[(orig, des)] = self._subproblem(orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate)
        return probs

    def _subproblem(self, orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/continuous/sub_{}_{}.mps'.format(self.ins_name, orig, des)
        if file_existence(dir_name) and (not regenerate):
            model = gp.read(dir_name)
            model = self._set_sub_params(model)
            lamb, theta, gamma = self._get_sub_variables(model, proj_edges, unsig_cross, G)
            model = self._store_sub_variables(model, lamb, theta, gamma)
        else:
            # initialize the sub-problem
            model = gp.Model('subproblem_{}-{}'.format(orig, des))
            model = self._set_sub_params(model)
            # parameters
            relevant_nodes, relevant_edges = self._get_relevant_nodes_edges(G, G_reverse, orig, des, M)
            proj_edges = set(proj_edges)
            theta_edges = [(i, j) for i, j in relevant_edges if (i, j) in proj_edges]
            relevant_edges = set(relevant_edges)
            gamma_unsig = [(i, j, g, h) for (i, j, g, h) in unsig_cross if (i, j) in relevant_edges]
            relevant_edges = list(relevant_edges)
            # add variables
            lamb = model.addVars(relevant_nodes, name='lambda', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            theta = model.addVars(theta_edges, name='theta', lb=0, vtype=gp.GRB.CONTINUOUS)
            gamma = model.addVars(gamma_unsig, name='gamma', lb=0, vtype=gp.GRB.CONTINUOUS)
            # add constraints - x
            for i, j in relevant_edges:
                exp = - lamb[j] + lamb[i]
                if (i, j) in theta:
                    exp -= theta[i, j]
                if i in unsig_set:
                    exp -= gp.quicksum(gamma[i, j, g, h] for g, h in G.edges(i) if self._diff_edges(i, j, g, h) and (g, h) in theta)
                model.addConstr(exp <= travel_time[i, j], name='x')
            # add constraints - z
            model.addConstr(-lamb[des] + lamb[orig] <= M, name='f')
            # store decision variables
            model = self._store_sub_variables(model, lamb, theta, gamma)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_sub_params(model):
        model.Params.outputFlag = 0
        return model

    @staticmethod
    def _get_sub_variables(model, proj_edges, unsig_cross, G):
        lamb = {i: model.getVarByName("lambda[{}]".format(i)) for i in G.nodes}
        theta = {(i, j): model.getVarByName("theta[{},{}]".format(i, j)) for (i, j) in proj_edges}
        gamma = {(i, j, g, h): model.getVarByName("gamma[{},{},{},{}]".format(i, j, g, h)) for (i, j, g, h) in unsig_cross}
        return lamb, theta, gamma

    @staticmethod
    def _store_sub_variables(model, lamb, theta, gamma):
        model._lamb = lamb
        model._theta = theta
        model._gamma = gamma
        return model

    @staticmethod
    def _get_relevant_nodes_edges(G, G_reverse, orig, des, M):
        reachable_orig = single_source_dijkstra_path_length(G=G, source=orig, cutoff=M, weight='time')
        reachable_des = single_source_dijkstra_path_length(G=G_reverse, source=des, cutoff=M, weight='time')
        relevant_nodes = set([])
        for node, val in reachable_orig.items():
            if (node in reachable_des) and (reachable_des[node] + val <= M):
                relevant_nodes.add(node)
                for (node, tnode) in G.out_edges(node):
                    relevant_nodes.add(tnode)
                for (fnode, node) in G.in_edges(node):
                    relevant_nodes.add(fnode)
        relevant_edges = []
        for fnode, tnode in G.edges():
            if (fnode in relevant_nodes) and (tnode in relevant_nodes):
                relevant_edges.append((fnode, tnode))
        return relevant_nodes, relevant_edges

    def _gen_pareto_problems(self, duals, n_nodes, projs, sig_costs, G):
        proj_edges = flatten(projs)
        proj_edge_set = self._proj_edge_set(proj_edges)
        unsig_set = set(sig_costs.keys())
        unsig_cross = [(i, j, g, h) for s in unsig_set for (i, j) in list(G.out_edges(s))
                       for (g, h) in list(G.in_edges(s)) + list(G.out_edges(s))
                       if self._diff_edges(i, j, g, h) and ('{}_{}'.format(g, h) in proj_edge_set)]
        print('generating pareto problem ...')
        probs = {}
        for orig, des in tqdm(duals):
            probs[orig, des] = self._pareto_problem(duals[orig, des], n_nodes, proj_edges, unsig_cross)
        return probs

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
        # penalties = model.getAttr('x', model._p)
        y_val = model.getAttr('x', model._y)
        new_projects = [i for i in model._y if y_val[i] >= 1 - 1e-5]
        if model._budget_sig > 0:
            s_val = model.getAttr('x', model._s)
            new_signals = [i for i in model._s if s_val[i] >= 1 - 1e-5]
        else:
            new_signals = []
        return new_projects, new_signals


class BendersBoostingSolver(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, beta_1,
              reg_factor=1, loss_bound=1,
              quiet=False, time_limit=None, regenerate=False, pareto=False,
              relax4cut=False, weighted=False, mip_gap=None):
        """
        solve the optimization problem
        :param args: instance arguments
        :param budget_proj:
        :param budget_sig:
        :param time_limit: solution time limit
        :return: lists of connected pairs, new projects, and new signals
        """
        if not quiet:
            print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T, M = self._args2params(args)
        weights = args['weights'] if 'weights' in args and weighted else {}
        beta = self._gen_betas(beta_1, T, M)
        if not quiet:
            print('The instance has {} od-pairs, {} projects, {} candiate intersections, {} edges, and {} nodes'.format(
                len(od_pairs), len(projs), len(sig_costs), len(G.edges()), n_nodes))
            print('compiling ...')
        tick = time.time()
        master = self._construct_master_problem(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig,
                                                T=T, beta=beta, weights=weights, proj_costs=proj_costs, sig_costs=sig_costs,
                                                time_limit=time_limit, regenerate=regenerate, quiet=quiet, mip_gap=mip_gap,
                                                reg_factor=1, loss_bound=loss_bound, args=args)
        if not quiet:
            master.update()
            print('Master problem has {} variables and {} constraints'.format(master.numVars, master.numConstrs))
        master._edge2proj = edge2proj
        master._T = T
        master._M = M
        master._y_hat, master._s_hat = self._find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs)
        master._sp = self._construct_subproblems(od_pairs, projs, sig_costs, G, travel_time, M, regenerate)
        if not quiet:
            master._sp[od_pairs[0]].update()
            print('A typical sub-problem problem has {} variables and {} constraints'.format(
                master._sp[od_pairs[0]].numVars, master._sp[od_pairs[0]].numConstrs))
        master._pareto = pareto
        master._cnt = 0
        if not quiet:
            print('  elapsed: {:.2f} sec'.format(time.time() - tick))
            print('solving ...')
        tick = time.time()
        if relax4cut:
            if not quiet:
                print('solving root relaxation for cheap cuts ...')
            cuts, v, y, s = self._relax4cut(master)
            cut_cnt = self._add_cheap_cuts(master, cuts, v, y, s)
            if not quiet:
                print('{} cuts found, {} of them added'.format(len(cuts), cut_cnt))
        n_cheap_cuts = cut_cnt if relax4cut else 0
        master.optimize(benders_cut)
        t_sol = time.time() - tick
        print('  obj val: {:.2f}'.format(master.objVal))
        print('  # of cuts added: {} = {} + {}'.format(n_cheap_cuts + master._cnt, n_cheap_cuts, master._cnt))
        print('  elapsed: {:.2f} sec'.format(t_sol))
        new_projects, new_signals = self._get_solution(master)
        return new_projects, new_signals, t_sol

    def _construct_master_problem(self, od_pairs, pop, budget_proj, budget_sig, proj_costs, sig_costs, beta, weights, T, time_limit, regenerate, quiet, mip_gap, reg_factor, loss_bound, args):
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
        dir_name = './prob/{}/models/continuous/master.mps'.format(self.ins_name)
        if file_existence(dir_name) and (not regenerate):
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            u, v, y, s = self._get_master_variables(model, od_pairs, list(proj_costs.keys()), list(sig_costs.keys()), budget_sig)
            # store variables
            model._u = u
            model._v = v
            model._y = y
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in list(sig_costs.keys())}
            model._budget_sig = budget_sig
            model = self._update_master_rhs(model, budget_proj, budget_sig)
        else:
            # get feature vector for out-of-sample prediction
            in_sample_feature = args['in_sample']
            out_of_sample_feature = args['out_of_sample']
            obj_weights = weights.copy()
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            # set parameters
            obj_weights = weights if len(weights) > 0 else {(orig, des): pop[des] for orig, des in od_pairs}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            dimensions = list(range(len(out_of_sample_feature)))
            # add variables
            u = model.addVars(od_pairs, name='u', vtype=gp.GRB.CONTINUOUS)
            v = model.addVars(od_pairs, name='v', vtype=gp.GRB.CONTINUOUS)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            w = model.addVars(dimensions, name='w', vtype=gp.GRB.CONTINUOUS, lb=-reg_factor, ub=reg_factor)
            a = model.addVars(dimensions, name='a', vtype=gp.GRB.CONTINUOUS)
            p = model.addVars(od_pairs, name='p', vtype=gp.GRB.CONTINUOUS)
            if budget_sig > 0:
                s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
            # add budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            if budget_sig > 0:
                model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # add time constraints
            model.addConstrs((u[orig, des] >= v[orig, des] - T for orig, des in od_pairs), name='time_exceeds')
            # regularization
            model.addConstrs((a[i] >= w[i] for i in dimensions), name='reg_pos')
            model.addConstrs((a[i] >= -w[i] for i in dimensions), name='reg_neg')
            model.addConstr(a.sum() <= reg_factor, name='reg')
            # training loss
            model.addConstrs((p[orig, des] >= w.prod(in_sample_feature[orig, des]) - beta[1] * v[orig, des] - (beta[2] - beta[1]) * u[orig, des]
                              for orig, des in od_pairs), name='loss_pos')
            model.addConstrs((p[orig, des] >= beta[1] * v[orig, des] + (beta[2] - beta[1]) * u[orig, des] - w.prod(in_sample_feature[orig, des])
                              for orig, des in od_pairs), name='loss_neg')
            model.addConstr(p.sum() <= loss_bound, name='loss_bound')
            # set objective
            obj = beta[1] * v.prod(obj_weights) + (beta[2] - beta[1]) * u.prod(obj_weights) + w.prod(out_of_sample_feature)
            model.setObjective(obj, gp.GRB.MINIMIZE)
            # add variable dicts
            model._u = u
            model._v = v
            model._y = y
            model._w = w
            model._budget_sig = budget_sig
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in signals}
            # write the problem to local drive
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_master_params(model, time_limit, quiet, mip_gap):
        model.Params.outputFlag = 0 if quiet else 1
        if time_limit:
            model.Params.timeLimit = time_limit
        if mip_gap:
            model.Params.mipGap = mip_gap
        model.Params.lazyConstraints = 1
        # model.Params.presolve = 0
        return model

    @staticmethod
    def _get_master_variables(model, od_pairs, projects, signals, budget_sig):
        u = gp.tupledict([((orig, des), model.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        v = gp.tupledict([((orig, des), model.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        y = gp.tupledict([(i, model.getVarByName("y[{}]".format(i))) for i in projects])
        if budget_sig > 0:
            s = gp.tupledict([(i, model.getVarByName("s[{}]".format(i))) for i in signals])
        else:
            s = []
        return u, v, y, s

    @staticmethod
    def _update_master_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

    @staticmethod
    def _find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs):
        """
        find the raltiave inner points (for the pareto cut problem)
        :param budget_proj:
        :param budget_sig:
        :param proj_costs:
        :param sig_costs:
        :return: dict for y, dict for s
        """
        n_proj = len(proj_costs)
        n_sig = len(sig_costs)
        y_hat = {idx: np.min([1, budget_proj / (2 * n_proj * val)]) for idx, val in proj_costs.items()}
        if budget_sig > 0:
            s_hat = {idx: np.min([1, budget_sig / (2 * n_sig * val)]) for idx, val in sig_costs.items()}
        else:
            s_hat = {idx: 0 for idx, val in sig_costs.items()}
        return y_hat, s_hat

    def _relax4cut(self, model):
        # initialize the relaxed problem
        relaxed = self._gen_relaxed_problem(model)
        relaxed.Params.outputFlag = 0
        relaxed.optimize(benders_cut_root_relax)
        v, y, s = self._get_relaxed_sol(relaxed)
        return relaxed._coeffs, v, y, s

    @staticmethod
    def _gen_relaxed_problem(model):
        model.update()
        relaxed = model.relax()
        relaxed._u = gp.tupledict([((orig, des), relaxed.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in model._u])
        relaxed._v = gp.tupledict([((orig, des), relaxed.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in model._v])
        relaxed._y = gp.tupledict([(i, relaxed.getVarByName("y[{}]".format(i))) for i in model._y])
        if model._budget_sig > 0:
            relaxed._s = gp.tupledict([(i, relaxed.getVarByName("s[{}]".format(i))) for i in model._s])
        else:
            relaxed._s = model._s.copy()
        # store parameters
        relaxed._M = model._M
        relaxed._edge2proj = model._edge2proj
        relaxed._cnt = 0
        relaxed._sp = model._sp
        relaxed._coeffs = []
        relaxed._y_hat = model._y_hat
        relaxed._s_hat = model._s_hat
        relaxed._budget_sig = model._budget_sig
        # add the dummy variable and the dummy constraint
        d = relaxed.addVar(name='dummy', vtype=gp.GRB.BINARY)
        relaxed.addConstr(d == 0, name='dummy')
        return relaxed

    @staticmethod
    def _get_relaxed_sol(model):
        if model.status == 9:
            return {}, {}, {}
        v = model.getAttr('x', model._v)
        y = model.getAttr('x', model._y)
        s = model.getAttr('x', model._s) if model._budget_sig > 0 else model._s.copy()
        return v, y, s

    def _add_cheap_cuts(self, model, cuts, v, y, s):
        cnt = 0
        not_opt = len(y) == 0
        for idx, cut in enumerate(cuts):
            if not_opt or (self._binding(cut, v, y, s)):
                cnt += 1
                rhs = cut['c'] + model._y.prod(cut['y'])
                if 's' in cut:
                    rhs += model._s.prod(cut['s'])
                model.addConstr(model._v[cut['od']] >= rhs, name='cheap{}'.format(idx))
        return cnt

    def _binding(self, cut, v, y, s):
        y_prod = np.sum([y[idx] * val for idx, val in cut['y'].items()])
        s_prod = np.sum([s[idx] * val for idx, val in cut['s'].items()]) if 's' in cut else 0
        rhs = cut['c'] + y_prod + s_prod
        return (v[cut['od']] > rhs - 1e-9) and (v[cut['od']] < rhs + 1e-9)

    def _construct_subproblems(self, od_pairs, projs, sig_costs, G, travel_time, M, regenerate):
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
        G_reverse = G.reverse()
        probs = {}
        for orig, des in tqdm(od_pairs):
            probs[(orig, des)] = self._subproblem(orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate)
        return probs

    def _subproblem(self, orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/continuous/sub_{}_{}.mps'.format(self.ins_name, orig, des)
        if file_existence(dir_name) and (not regenerate):
            model = gp.read(dir_name)
            model = self._set_sub_params(model)
            lamb, theta, gamma = self._get_sub_variables(model, proj_edges, unsig_cross, G)
            model = self._store_sub_variables(model, lamb, theta, gamma)
        else:
            # initialize the sub-problem
            model = gp.Model('subproblem_{}-{}'.format(orig, des))
            model = self._set_sub_params(model)
            # parameters
            relevant_nodes, relevant_edges = self._get_relevant_nodes_edges(G, G_reverse, orig, des, M)
            proj_edges = set(proj_edges)
            theta_edges = [(i, j) for i, j in relevant_edges if (i, j) in proj_edges]
            relevant_edges = set(relevant_edges)
            gamma_unsig = [(i, j, g, h) for (i, j, g, h) in unsig_cross if (i, j) in relevant_edges]
            relevant_edges = list(relevant_edges)
            # add variables
            lamb = model.addVars(relevant_nodes, name='lambda', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            theta = model.addVars(theta_edges, name='theta', lb=0, vtype=gp.GRB.CONTINUOUS)
            gamma = model.addVars(gamma_unsig, name='gamma', lb=0, vtype=gp.GRB.CONTINUOUS)
            # add constraints - x
            for i, j in relevant_edges:
                exp = - lamb[j] + lamb[i]
                if (i, j) in theta:
                    exp -= theta[i, j]
                if i in unsig_set:
                    exp -= gp.quicksum(gamma[i, j, g, h] for g, h in G.edges(i) if self._diff_edges(i, j, g, h) and (g, h) in theta)
                model.addConstr(exp <= travel_time[i, j], name='x')
            # add constraints - z
            model.addConstr(-lamb[des] + lamb[orig] <= M, name='f')
            # store decision variables
            model = self._store_sub_variables(model, lamb, theta, gamma)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_sub_params(model):
        model.Params.outputFlag = 0
        return model

    @staticmethod
    def _get_sub_variables(model, proj_edges, unsig_cross, G):
        lamb = {i: model.getVarByName("lambda[{}]".format(i)) for i in G.nodes}
        theta = {(i, j): model.getVarByName("theta[{},{}]".format(i, j)) for (i, j) in proj_edges}
        gamma = {(i, j, g, h): model.getVarByName("gamma[{},{},{},{}]".format(i, j, g, h)) for (i, j, g, h) in unsig_cross}
        return lamb, theta, gamma

    @staticmethod
    def _store_sub_variables(model, lamb, theta, gamma):
        model._lamb = lamb
        model._theta = theta
        model._gamma = gamma
        return model

    @staticmethod
    def _get_relevant_nodes_edges(G, G_reverse, orig, des, M):
        reachable_orig = single_source_dijkstra_path_length(G=G, source=orig, cutoff=M, weight='time')
        reachable_des = single_source_dijkstra_path_length(G=G_reverse, source=des, cutoff=M, weight='time')
        relevant_nodes = set([])
        for node, val in reachable_orig.items():
            if (node in reachable_des) and (reachable_des[node] + val <= M):
                relevant_nodes.add(node)
                for (node, tnode) in G.out_edges(node):
                    relevant_nodes.add(tnode)
                for (fnode, node) in G.in_edges(node):
                    relevant_nodes.add(fnode)
        relevant_edges = []
        for fnode, tnode in G.edges():
            if (fnode in relevant_nodes) and (tnode in relevant_nodes):
                relevant_edges.append((fnode, tnode))
        return relevant_nodes, relevant_edges

    def _gen_pareto_problems(self, duals, n_nodes, projs, sig_costs, G):
        proj_edges = flatten(projs)
        proj_edge_set = self._proj_edge_set(proj_edges)
        unsig_set = set(sig_costs.keys())
        unsig_cross = [(i, j, g, h) for s in unsig_set for (i, j) in list(G.out_edges(s))
                       for (g, h) in list(G.in_edges(s)) + list(G.out_edges(s))
                       if self._diff_edges(i, j, g, h) and ('{}_{}'.format(g, h) in proj_edge_set)]
        print('generating pareto problem ...')
        probs = {}
        for orig, des in tqdm(duals):
            probs[orig, des] = self._pareto_problem(duals[orig, des], n_nodes, proj_edges, unsig_cross)
        return probs

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
        # penalties = model.getAttr('x', model._p)
        y_val = model.getAttr('x', model._y)
        new_projects = [i for i in model._y if y_val[i] >= 1 - 1e-5]
        if model._budget_sig > 0:
            s_val = model.getAttr('x', model._s)
            new_signals = [i for i in model._s if s_val[i] >= 1 - 1e-5]
        else:
            new_signals = []
        return new_projects, new_signals


class BendersSolverEquity(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, beta_1, quiet=False, fixed_project=[], equity_parameter=0.5,
              time_limit=None, regenerate=False, pareto=False, relax4cut=False, weighted=False, mip_gap=None,
              equity_type='equality'):
        """
        solve the optimization problem
        :param args: instance arguments
        :param budget_proj:
        :param budget_sig:
        :param time_limit: solution time limit
        :return: lists of connected pairs, new projects, and new signals
        """
        if not quiet:
            print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T, M = self._args2params(args)
        weights = args['weights'] if 'weights' in args and weighted else {}
        equity_weights = self._gen_equity_weights(args=args, od_pairs=od_pairs, pop=pop)
        beta = self._gen_betas(beta_1, T, M)
        if not quiet:
            print('The instance has {} od-pairs, {} projects, {} candiate intersections, {} edges, and {} nodes'.format(
                len(od_pairs), len(projs), len(sig_costs), len(G.edges()), n_nodes))
            print('Fixed project:', fixed_project)
            print('compiling ...')
        tick = time.time()
        master = self._construct_master_problem(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig,
                                                T=T, beta=beta, weights=weights, proj_costs=proj_costs, sig_costs=sig_costs,
                                                time_limit=time_limit, regenerate=regenerate, quiet=quiet, mip_gap=mip_gap,
                                                fixed_project=fixed_project, equity_weights=equity_weights,
                                                equity_parameter=equity_parameter, equity_type=equity_type, M=M)
        if not quiet:
            master.update()
            print('Master problem has {} variables and {} constraints'.format(master.numVars, master.numConstrs))
        master._edge2proj = edge2proj
        master._T = T
        master._M = M
        master._y_hat, master._s_hat = self._find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs)
        master._sp = self._construct_subproblems(od_pairs, projs, sig_costs, G, travel_time, M, regenerate)
        if not quiet:
            master._sp[od_pairs[0]].update()
            print('A typical sub-problem problem has {} variables and {} constraints'.format(
                master._sp[od_pairs[0]].numVars, master._sp[od_pairs[0]].numConstrs))
        master._pareto = pareto
        master._cnt = 0
        if not quiet:
            print('  elapsed: {:.2f} sec'.format(time.time() - tick))
            print('solving ...')
        tick = time.time()
        if relax4cut:
            if not quiet:
                print('solving root relaxation for cheap cuts ...')
            cuts, v, y, s = self._relax4cut(master)
            cut_cnt = self._add_cheap_cuts(master, cuts, v, y, s)
            if not quiet:
                print('{} cuts found, {} of them added'.format(len(cuts), cut_cnt))
        n_cheap_cuts = cut_cnt if relax4cut else 0
        master.optimize(benders_cut)
        t_sol = time.time() - tick
        print('  obj val: {:.2f}'.format(master.objVal))
        print('  # of cuts added: {} = {} + {}'.format(n_cheap_cuts + master._cnt, n_cheap_cuts, master._cnt))
        print('  elapsed: {:.2f} sec'.format(t_sol))
        new_projects, new_signals = self._get_solution(master)
        # print(master._efficiency.X)
        # print(master._equity.X)
        print(master.getAttr('x', master._U))
        # print(master.getAttr('x', master._V))
        return new_projects, new_signals, t_sol

    def _construct_master_problem(self, od_pairs, pop, budget_proj, budget_sig, proj_costs, sig_costs, beta, weights,
                                  T, M, time_limit, regenerate, quiet, mip_gap, fixed_project, equity_weights,
                                  equity_parameter, equity_type):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/continuous/master.mps'.format(self.ins_name)
        if file_existence(dir_name) and (not regenerate):
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            u, v, y, s = self._get_master_variables(model, od_pairs, list(proj_costs.keys()), list(sig_costs.keys()), budget_sig)
            # store variables
            model._u = u
            model._v = v
            model._y = y
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in list(sig_costs.keys())}
            model._budget_sig = budget_sig
            model = self._update_master_rhs(model, budget_proj, budget_sig)
        else:
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_master_params(model, time_limit, quiet, mip_gap)
            # set parameters
            obj_weights = weights if len(weights) > 0 else {(orig, des): pop[des] for orig, des in od_pairs}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            ses_groups = list(range(1, 6))
            ses_group_combos = [(i, j) for i in range(1, 6) for j in range(1, 6) if i != j]
            # add variables
            u = model.addVars(od_pairs, name='u', vtype=gp.GRB.CONTINUOUS, ub=M-T)
            v = model.addVars(od_pairs, name='v', vtype=gp.GRB.CONTINUOUS, ub=M)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            # add budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            if budget_sig > 0:
                s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
                model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # fix some project if necessary
            if len(fixed_project):
                model.addConstrs((y[i] == 1 for i in fixed_project), name='fixed_project')
            # add time constraints
            model.addConstrs((u[orig, des] == v[orig, des] - T for orig, des in od_pairs), name='time_exceeds')
            # add equity constraints
            if equity_type == 'equality':
                # V = model.addVars(ses_group_combos, name='V', vtype=gp.GRB.CONTINUOUS, lb=0, ub=1)
                # model.addConstrs((V[i, j] >= U[i] - U[j] for i, j in ses_group_combos), name='diff_pos')
                # model.addConstrs((V[i, j] >= U[j] - U[i] for i, j in ses_group_combos), name='diff_neg')
                # model.addConstr(equity_obj == V.sum() / 5, name='equity_obj')
                # U = model.addVars(ses_groups, name='U', vtype=gp.GRB.CONTINUOUS, lb=0, ub=1)
                # model.addConstrs((U[i] ==
                #                   np.sum(list(equity_weights[i].values())[:-1])
                #                   - beta[1] * v.prod(equity_weights[i])
                #                   - (beta[2] - beta[1]) * u.prod(equity_weights[i])
                #                   + equity_weights[i][0] for i in ses_groups), name='group_utility')
                # efficiency_obj = model.addVar(name='efficiency')
                # equity_obj = model.addVar(name='equity')
                # omega = model.addVar(name='omega', ub=1, lb=0, vtype=gp.GRB.CONTINUOUS)
                # model.addConstr(omega >= U[ses_groups[0]] - U[ses_groups[-1]])
                # model.addConstr(omega >= U[ses_groups[-1]] - U[ses_groups[0]])
                # model.addConstr(equity_obj == omega, name='equity_obj')
                # model.addConstr(efficiency_obj == (beta[1] * v.prod(obj_weights) + (beta[2] - beta[1]) * u.prod(obj_weights)) / D, name='efficiency_obj')
                # model.setObjective((1 - equity_parameter) * efficiency_obj + equity_parameter * equity_obj, gp.GRB.MINIMIZE)
                # equity_obj = V.sum() / 5
                weight_1 = equity_weights[1]
                weight_5 = equity_weights[5]
                big_w = np.sum(list(obj_weights.values()))
                weight = {key: (1 - equity_parameter) * val / big_w + equity_parameter * (weight_5[key] - weight_1[key])
                          for key, val in obj_weights.items()}
                weight = {key: np.max([val, 0]) for key, val in weight.items()}
                print(equity_parameter)
                print(weight)
                model.setObjective(beta[1] * v.prod(weight) + (beta[2] - beta[1]) * u.prod(weight), gp.GRB.MINIMIZE)
            elif equity_type == 'rawlsian':
                U = model.addVars(ses_groups, name='U', vtype=gp.GRB.CONTINUOUS)
                model.addConstrs((U[i] ==
                                  np.sum(list(equity_weights[i].values())[:-1])
                                  - beta[1] * v.prod(equity_weights[i])
                                  - (beta[2] - beta[1]) * u.prod(equity_weights[i])
                                  + equity_weights[i][0] for i in ses_groups), name='group_utility')
                efficiency_obj = model.addVar(name='efficiency', lb=-gp.GRB.INFINITY, ub=0)
                equity_obj = model.addVar(name='equity', lb=-gp.GRB.INFINITY, ub=0)
                omega = model.addVar(name='omega', vtype=gp.GRB.CONTINUOUS)
                model.addConstrs((omega <= U[i] for i in ses_groups), name='minimum_group')
                model.addConstr(equity_obj == - omega * len(ses_groups), name='equity_obj')
                model.addConstr(efficiency_obj == - U.sum(), name='efficiency_obj')
                model.setObjective((1 - equity_parameter) * efficiency_obj + equity_parameter * equity_obj, gp.GRB.MINIMIZE)
                # equity_obj = - omega
            elif equity_type == 'nash':
                U = model.addVars(ses_groups, name='U', vtype=gp.GRB.CONTINUOUS, lb=0, ub=1)
                model.addConstrs((U[i] ==
                                  np.sum(list(equity_weights[i].values())[:-1])
                                  - beta[1] * v.prod(equity_weights[i])
                                  - (beta[2] - beta[1]) * u.prod(equity_weights[i])
                                  + equity_weights[i][0] for i in ses_groups), name='group_utility')
                efficiency_obj = model.addVar(name='efficiency')
                equity_obj = model.addVar(name='equity')
                V = model.addVars(ses_groups, name='V', ub=0, lb=-gp.GRB.INFINITY)
                break_points = np.arange(0.1, 1, 0.1)
                model.addConstrs((V[i] <= (1/m) * U[i] + np.log(m) - 1
                                  for i in ses_groups for m in break_points), 'log_approx')
                model.addConstr(equity_obj == - V.sum(), name='equity_obj')
                model.addConstr(efficiency_obj == (beta[1] * v.prod(obj_weights) + (beta[2] - beta[1]) * u.prod(obj_weights)), name='efficiency_obj')
                model.setObjective((1 - equity_parameter) * efficiency_obj + equity_parameter * equity_obj, gp.GRB.MINIMIZE)
            else:
                raise ValueError('Equity type {} is not specified'.format(equity_type))

            # set objective
            # efficiency_obj = (beta[1] * v.prod(obj_weights) + (beta[2] - beta[1]) * u.prod(obj_weights)) / D
            # add variable dicts
            model._u = u
            model._v = v
            model._y = y
            model._U = U
            # model._V = V
            # model._efficiency = efficiency_obj
            # model._equity = equity_obj
            model._budget_sig = budget_sig
            if budget_sig > 0:
                model._s = s
            else:
                model._s = {idx: 0 for idx in signals}
            # write the problem to local drive
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_master_params(model, time_limit, quiet, mip_gap):
        model.Params.outputFlag = 0 if quiet else 1
        if time_limit:
            model.Params.timeLimit = time_limit
        if mip_gap:
            model.Params.mipGap = mip_gap
        model.Params.lazyConstraints = 1
        # model.Params.presolve = 0
        return model

    @staticmethod
    def _get_master_variables(model, od_pairs, projects, signals, budget_sig):
        u = gp.tupledict([((orig, des), model.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        v = gp.tupledict([((orig, des), model.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in od_pairs])
        y = gp.tupledict([(i, model.getVarByName("y[{}]".format(i))) for i in projects])
        if budget_sig > 0:
            s = gp.tupledict([(i, model.getVarByName("s[{}]".format(i))) for i in signals])
        else:
            s = []
        return u, v, y, s

    @staticmethod
    def _update_master_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

    @staticmethod
    def _find_inner_points(budget_proj, budget_sig, proj_costs, sig_costs):
        """
        find the raltiave inner points (for the pareto cut problem)
        :param budget_proj:
        :param budget_sig:
        :param proj_costs:
        :param sig_costs:
        :return: dict for y, dict for s
        """
        n_proj = len(proj_costs)
        n_sig = len(sig_costs)
        y_hat = {idx: np.min([1, budget_proj / (2 * n_proj * val)]) for idx, val in proj_costs.items()}
        if budget_sig > 0:
            s_hat = {idx: np.min([1, budget_sig / (2 * n_sig * val)]) for idx, val in sig_costs.items()}
        else:
            s_hat = {idx: 0 for idx, val in sig_costs.items()}
        return y_hat, s_hat

    def _relax4cut(self, model):
        # initialize the relaxed problem
        relaxed = self._gen_relaxed_problem(model)
        relaxed.Params.outputFlag = 0
        relaxed.optimize(benders_cut_root_relax)
        v, y, s = self._get_relaxed_sol(relaxed)
        return relaxed._coeffs, v, y, s

    @staticmethod
    def _gen_relaxed_problem(model):
        model.update()
        relaxed = model.relax()
        relaxed._u = gp.tupledict([((orig, des), relaxed.getVarByName("u[{},{}]".format(orig, des))) for (orig, des) in model._u])
        relaxed._v = gp.tupledict([((orig, des), relaxed.getVarByName("v[{},{}]".format(orig, des))) for (orig, des) in model._v])
        relaxed._y = gp.tupledict([(i, relaxed.getVarByName("y[{}]".format(i))) for i in model._y])
        if model._budget_sig > 0:
            relaxed._s = gp.tupledict([(i, relaxed.getVarByName("s[{}]".format(i))) for i in model._s])
        else:
            relaxed._s = model._s.copy()
        # store parameters
        relaxed._M = model._M
        relaxed._edge2proj = model._edge2proj
        relaxed._cnt = 0
        relaxed._sp = model._sp
        relaxed._coeffs = []
        relaxed._y_hat = model._y_hat
        relaxed._s_hat = model._s_hat
        relaxed._budget_sig = model._budget_sig
        # add the dummy variable and the dummy constraint
        d = relaxed.addVar(name='dummy', vtype=gp.GRB.BINARY)
        relaxed.addConstr(d == 0, name='dummy')
        return relaxed

    @staticmethod
    def _get_relaxed_sol(model):
        if model.status == 9:
            return {}, {}, {}
        v = model.getAttr('x', model._v)
        y = model.getAttr('x', model._y)
        s = model.getAttr('x', model._s) if model._budget_sig > 0 else model._s.copy()
        return v, y, s

    def _add_cheap_cuts(self, model, cuts, v, y, s):
        cnt = 0
        not_opt = len(y) == 0
        for idx, cut in enumerate(cuts):
            if not_opt or (self._binding(cut, v, y, s)):
                cnt += 1
                rhs = cut['c'] + model._y.prod(cut['y'])
                if 's' in cut:
                    rhs += model._s.prod(cut['s'])
                model.addConstr(model._v[cut['od']] >= rhs, name='cheap{}'.format(idx))
        return cnt

    def _binding(self, cut, v, y, s):
        y_prod = np.sum([y[idx] * val for idx, val in cut['y'].items()])
        s_prod = np.sum([s[idx] * val for idx, val in cut['s'].items()]) if 's' in cut else 0
        rhs = cut['c'] + y_prod + s_prod
        return (v[cut['od']] > rhs - 1e-9) and (v[cut['od']] < rhs + 1e-9)

    def _construct_subproblems(self, od_pairs, projs, sig_costs, G, travel_time, M, regenerate):
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
        G_reverse = G.reverse()
        probs = {}
        for orig, des in tqdm(od_pairs):
            probs[(orig, des)] = self._subproblem(orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate)
        return probs

    def _subproblem(self, orig, des, G, G_reverse, proj_edges, unsig_set, unsig_cross, travel_time, M, regenerate):
        # check if the problem has been generated or not
        dir_name = './prob/{}/models/continuous/sub_{}_{}.mps'.format(self.ins_name, orig, des)
        if file_existence(dir_name) and (not regenerate):
            model = gp.read(dir_name)
            model = self._set_sub_params(model)
            lamb, theta, gamma = self._get_sub_variables(model, proj_edges, unsig_cross, G)
            model = self._store_sub_variables(model, lamb, theta, gamma)
        else:
            # initialize the sub-problem
            model = gp.Model('subproblem_{}-{}'.format(orig, des))
            model = self._set_sub_params(model)
            # parameters
            relevant_nodes, relevant_edges = self._get_relevant_nodes_edges(G, G_reverse, orig, des, M)
            proj_edges = set(proj_edges)
            theta_edges = [(i, j) for i, j in relevant_edges if (i, j) in proj_edges]
            relevant_edges = set(relevant_edges)
            gamma_unsig = [(i, j, g, h) for (i, j, g, h) in unsig_cross if (i, j) in relevant_edges]
            relevant_edges = list(relevant_edges)
            # add variables
            lamb = model.addVars(relevant_nodes, name='lambda', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
            theta = model.addVars(theta_edges, name='theta', lb=0, vtype=gp.GRB.CONTINUOUS)
            gamma = model.addVars(gamma_unsig, name='gamma', lb=0, vtype=gp.GRB.CONTINUOUS)
            # add constraints - x
            for i, j in relevant_edges:
                exp = - lamb[j] + lamb[i]
                if (i, j) in theta:
                    exp -= theta[i, j]
                if i in unsig_set:
                    exp -= gp.quicksum(gamma[i, j, g, h] for g, h in G.edges(i) if self._diff_edges(i, j, g, h) and (g, h) in theta)
                model.addConstr(exp <= travel_time[i, j], name='x')
            # add constraints - z
            model.addConstr(-lamb[des] + lamb[orig] <= M, name='f')
            # store decision variables
            model = self._store_sub_variables(model, lamb, theta, gamma)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_sub_params(model):
        model.Params.outputFlag = 0
        return model

    @staticmethod
    def _get_sub_variables(model, proj_edges, unsig_cross, G):
        lamb = {i: model.getVarByName("lambda[{}]".format(i)) for i in G.nodes}
        theta = {(i, j): model.getVarByName("theta[{},{}]".format(i, j)) for (i, j) in proj_edges}
        gamma = {(i, j, g, h): model.getVarByName("gamma[{},{},{},{}]".format(i, j, g, h)) for (i, j, g, h) in unsig_cross}
        return lamb, theta, gamma

    @staticmethod
    def _store_sub_variables(model, lamb, theta, gamma):
        model._lamb = lamb
        model._theta = theta
        model._gamma = gamma
        return model

    @staticmethod
    def _get_relevant_nodes_edges(G, G_reverse, orig, des, M):
        reachable_orig = single_source_dijkstra_path_length(G=G, source=orig, cutoff=M, weight='time')
        reachable_des = single_source_dijkstra_path_length(G=G_reverse, source=des, cutoff=M, weight='time')
        relevant_nodes = set([])
        for node, val in reachable_orig.items():
            if (node in reachable_des) and (reachable_des[node] + val <= M):
                relevant_nodes.add(node)
                for (node, tnode) in G.out_edges(node):
                    relevant_nodes.add(tnode)
                for (fnode, node) in G.in_edges(node):
                    relevant_nodes.add(fnode)
        relevant_edges = []
        for fnode, tnode in G.edges():
            if (fnode in relevant_nodes) and (tnode in relevant_nodes):
                relevant_edges.append((fnode, tnode))
        return relevant_nodes, relevant_edges

    def _gen_pareto_problems(self, duals, n_nodes, projs, sig_costs, G):
        proj_edges = flatten(projs)
        proj_edge_set = self._proj_edge_set(proj_edges)
        unsig_set = set(sig_costs.keys())
        unsig_cross = [(i, j, g, h) for s in unsig_set for (i, j) in list(G.out_edges(s))
                       for (g, h) in list(G.in_edges(s)) + list(G.out_edges(s))
                       if self._diff_edges(i, j, g, h) and ('{}_{}'.format(g, h) in proj_edge_set)]
        print('generating pareto problem ...')
        probs = {}
        for orig, des in tqdm(duals):
            probs[orig, des] = self._pareto_problem(duals[orig, des], n_nodes, proj_edges, unsig_cross)
        return probs

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
        # penalties = model.getAttr('x', model._p)
        y_val = model.getAttr('x', model._y)
        new_projects = [i for i in model._y if y_val[i] >= 1 - 1e-5]
        if model._budget_sig > 0:
            s_val = model.getAttr('x', model._s)
            new_signals = [i for i in model._s if s_val[i] >= 1 - 1e-5]
        else:
            new_signals = []
        return new_projects, new_signals

    @staticmethod
    def _gen_equity_weights(args, od_pairs, pop):
        neighbors = args['neighbors']
        od_init = args['od_pairs_init']
        acc_tot, _ = load_file('./data/on_marg_index/tot_acc.pkl')
        acc_curr, _ = load_file('./data/on_marg_index/curr_acc.pkl')
        marg_groups, _ = load_file('./data/on_marg_index/marg_groups.pkl')
        node2group, _ = load_file('./data/on_marg_index/node2group.pkl')
        weights = {idx: {(orig, des): 0 for (orig, des) in od_pairs} for idx in range(1, 6)}
        for idx, pair in enumerate(od_init):
            orig, des = pair
            g = node2group[orig]
            for n in neighbors[idx]:
                weights[g][od_init[n]] += pop[des] / len(neighbors[idx])  # / len(marg_groups[g]) / acc_tot[orig]
        # calculate constant terms
        for g, nodes in marg_groups.items():
            # weights[g][0] = np.mean([acc_curr[orig] / acc_tot[orig] for orig in nodes if orig in acc_tot])
            weights[g][0] = np.mean([acc_curr[orig] for orig in nodes])
        return weights

