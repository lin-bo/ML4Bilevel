#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin
from solver.route_choice.abstract_solver import AbstractSolver
import gurobipy as gp
import numpy as np
import time
from utils.functions import des2od


class BilevelSolver(AbstractSolver):

    def solve(self, args, budget, weighted=False, n_breakpoints=5):
        # extract information from the argument
        od_pairs, destination, pop, seg2idx, v_bar, beta, G, n_nodes, projs, proj_costs, \
        sig_costs, travel_time, edge2proj, segidx2proj, segs = self._args2params(args)
        bp = [i/(n_breakpoints + 1) for i in range(1, n_breakpoints + 1)]
        weight = args['weights'] if 'weights' in args and weighted else {}
        model = self._construct_mip(proj_costs=proj_costs, seg2idx=seg2idx, od_pairs=od_pairs,
                                    destination=destination, bp=bp, budget=budget, segidx2proj=segidx2proj, v_bar=v_bar,
                                    beta=beta, segs=segs, pop=pop, weight=weight)
        tick = time.time()
        model.optimize()
        t_sol = time.time() - tick
        # retrieve the opt solution
        new_projects = self._get_solution(model)
        obj_approx = model.objVal
        # print('# of OD pairs:', len(od_pairs), 'total pop:', np.sum([pop[des] for _, des in od_pairs]))
        # print('New projects:', new_projects)
        # print('Approx obj:', model.objVal)
        return obj_approx, t_sol, new_projects

    def pwo_solve(self, args, budget, c_feature_dict, o_feature_dict, lambd=1, n_breakpoints=5, L_bar=10):
        # extract information from the argument
        od_pairs, destination, pop, seg2idx, v_bar, beta, G, n_nodes, projs, proj_costs, \
        sig_costs, travel_time, edge2proj, segidx2proj, segs = self._args2params(args)
        # set parameters
        bp = [i / (n_breakpoints + 1) for i in range(1, n_breakpoints + 1)]
        # construct
        re_solve = True
        while re_solve:
            model = self._construct_pwo_mip(proj_costs=proj_costs, seg2idx=seg2idx, od_pairs=od_pairs,
                                            destination=destination, bp=bp, budget=budget, segidx2proj=segidx2proj, v_bar=v_bar,
                                            beta=beta, segs=segs, pop=pop, c_feature_dict=c_feature_dict,
                                            o_feature_dict=o_feature_dict, lambd=lambd, L_bar=L_bar)
            tick = time.time()
            model.optimize()
            t_sol = time.time() - tick
            if model.status == 3:
                L_bar += len(od_pairs) * 0.1
            else:
                re_solve = False
        # retrieve the opt solution
        new_projects = self._get_solution(model)
        obj_approx = model.objVal
        # print('# of OD pairs:', len(od_pairs), 'total pop:', np.sum([pop[des] for _, des in od_pairs]))
        # print('New projects:', new_projects)
        # print('Approx obj:', model.objVal)
        return obj_approx, t_sol, new_projects

    def _construct_mip(self, proj_costs, budget, seg2idx, od_pairs, destination, bp, segidx2proj, v_bar, beta, segs, pop, weight):
        # set parameters
        projects = list(proj_costs.keys())
        segments = list(range(len(seg2idx)))
        route_id = list(range(3))
        od_x_route = [(orig, des, i) for orig, des in od_pairs for i in route_id]
        zeta_idx = [(orig, des, i, l) for orig, des in od_pairs for i in route_id for l in destination[orig][des][i]]
        M = list(range(len(bp)))
        obj_weight = weight if len(weight) > 0 else {(orig, des): pop[des] for orig, des in od_pairs}
        # intialize the problem
        model = gp.Model('route_choice')
        model.Params.outputFlag = 0
        # add variables
        z = model.addVars(projects, name='z', vtype=gp.GRB.BINARY)                  # project construction
        y = model.addVars(segments, name='y', lb=0, ub=1, vtype=gp.GRB.CONTINUOUS)  # utility function approx
        zeta = model.addVars(zeta_idx, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS)         #
        p = model.addVars(od_x_route)                                       # route probability
        theta = model.addVars(od_x_route, M)                                # dual variable theta
        gamma = model.addVars(od_pairs, name='gamma', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        w = model.addVars(od_x_route, name='w', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        phi = model.addVars(od_x_route, name='phi', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        # add constraints
        model.addConstr(z.prod(proj_costs) <= budget, name='project_budget')
        model.addConstrs((y[l] <= z[p] for l in segments for p in segidx2proj[l]
                          if len(segidx2proj[l]) <= 2), name='utility_approx_one')
        model.addConstrs((y[l] <= y[seg2idx[segs[l][1:]]] for l in segments
                          if len(segidx2proj[l]) > 2), name='utility_approx_multi_neg')
        model.addConstrs((y[l] <= y[seg2idx[segs[l][:-1]]] for l in segments
                          if len(segidx2proj[l]) > 2), name='utility_approx_multi_pos')
        model.addConstrs((gp.quicksum(p[orig, des, i] for i in route_id) == 1 for (orig, des) in od_pairs), name='prob_sum')
        model.addConstrs((w[orig, des, i] >= p[orig, des, i] * (np.log(bp[m]) + 1) - bp[m]
                          for (orig, des, i) in od_x_route for m in M), name='convex_approx')
        model.addConstrs((gamma[orig, des] - gp.quicksum((np.log(bp[m]) + 1) * theta[orig, des, i, m] for m in M)
                          <= v_bar[orig][des][i] - gp.quicksum(beta[len(segs[l])] * y[l] for l in destination[orig][des][i])
                          for (orig, des, i) in od_x_route), name='dual_first')
        model.addConstrs((gp.quicksum(theta[orig, des, i, m] for m in M) == 1
                         for orig, des in od_pairs for i in route_id), name='dual_second')
        model.addConstr(w.sum() - phi.sum() + gp.quicksum(p[orig, des, i] * v_bar[orig][des][i] for (orig, des, i) in od_x_route)
                        == gamma.sum() - gp.quicksum(bp[m] * theta[orig, des, i, m]
                                                     for (orig, des, i) in od_x_route for m in M),
                        name='primal_dual_equal')
        model.addConstrs((phi[orig, des, i] == gp.quicksum(beta[len(segs[l])-1] * zeta[orig, des, i, l]
                                                           for l in destination[orig][des][i])
                          for (orig, des, i) in od_x_route), name='route_choice_approx')
        model.addConstrs((zeta[orig, des, i, l] <= p[orig, des, i]
                          for (orig, des, i) in od_x_route for l in destination[orig][des][i]),
                         name='route_choice_approx_p')
        model.addConstrs((zeta[orig, des, i, l] <= y[l]
                          for (orig, des, i) in od_x_route for l in destination[orig][des][i]),
                         name='route_choice_approx_y')
        # add objective
        obj = gp.quicksum(obj_weight[orig, des] * (phi[orig, des, i] - p[orig, des, i] * v_bar[orig][des][i])
                          for (orig, des, i) in od_x_route)
        model.setObjective(obj, gp.GRB.MAXIMIZE)
        model.update()
        # store variables
        model._z = z
        model._zeta = zeta
        model._p = p
        model._phi = phi
        return model

    def _construct_pwo_mip(self, proj_costs, budget, seg2idx, od_pairs, destination, bp, segidx2proj, v_bar, beta, segs, pop, c_feature_dict, o_feature_dict, lambd, L_bar):
        # set parameters
        projects = list(proj_costs.keys())
        segments = list(range(len(seg2idx)))
        route_id = list(range(3))
        od_x_route = [(orig, des, i) for orig, des in od_pairs for i in route_id]
        zeta_idx = [(orig, des, i, l) for orig, des in od_pairs for i in route_id for l in destination[orig][des][i]]
        M = list(range(len(bp)))
        dim = len(o_feature_dict[list(o_feature_dict.keys())[0]])
        D = list(range(dim))
        # intialize the problem
        model = gp.Model('route_choice')
        model.Params.outputFlag = 0
        # add variables
        z = model.addVars(projects, name='z', vtype=gp.GRB.BINARY)  # project construction
        y = model.addVars(segments, name='y', lb=0, ub=1, vtype=gp.GRB.CONTINUOUS)  # utility function approx
        zeta = model.addVars(zeta_idx, lb=0, ub=1, vtype=gp.GRB.CONTINUOUS)  #
        p = model.addVars(od_x_route)  # route probability
        theta = model.addVars(od_x_route, M)  # dual variable theta
        gamma = model.addVars(od_pairs, name='gamma', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        w = model.addVars(od_x_route, name='w', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        phi = model.addVars(od_x_route, name='phi', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        omega = model.addVars(D, name='omega', lb=-gp.GRB.INFINITY, ub=gp.GRB.INFINITY)
        a = model.addVars(D, name='a')
        e = model.addVars(od_pairs, name='e')
        # add constraints
        model.addConstr(z.prod(proj_costs) <= budget, name='project_budget')
        model.addConstrs((y[l] <= z[p] for l in segments for p in segidx2proj[l]
                          if len(segidx2proj[l]) <= 2), name='utility_approx_one')
        model.addConstrs((y[l] <= y[seg2idx[segs[l][1:]]] for l in segments
                          if len(segidx2proj[l]) > 2), name='utility_approx_multi_neg')
        model.addConstrs((y[l] <= y[seg2idx[segs[l][:-1]]] for l in segments
                          if len(segidx2proj[l]) > 2), name='utility_approx_multi_pos')
        model.addConstrs((gp.quicksum(p[orig, des, i] for i in route_id) == 1 for (orig, des) in od_pairs),
                         name='prob_sum')
        model.addConstrs((w[orig, des, i] >= p[orig, des, i] * (np.log(bp[m]) + 1) - bp[m]
                          for (orig, des, i) in od_x_route for m in M), name='convex_approx')
        model.addConstrs((gamma[orig, des] - gp.quicksum((np.log(bp[m]) + 1) * theta[orig, des, i, m] for m in M)
                          <= v_bar[orig][des][i] - gp.quicksum(
            beta[len(segs[l])] * y[l] for l in destination[orig][des][i])
                          for (orig, des, i) in od_x_route), name='dual_first')
        model.addConstrs((gp.quicksum(theta[orig, des, i, m] for m in M) == 1
                          for orig, des in od_pairs for i in route_id), name='dual_second')
        model.addConstr(
            w.sum() - phi.sum() + gp.quicksum(p[orig, des, i] * v_bar[orig][des][i] for (orig, des, i) in od_x_route)
            == gamma.sum() - gp.quicksum(bp[m] * theta[orig, des, i, m]
                                         for (orig, des, i) in od_x_route for m in M),
            name='primal_dual_equal')
        model.addConstrs((phi[orig, des, i] == gp.quicksum(beta[len(segs[l]) - 1] * zeta[orig, des, i, l]
                                                           for l in destination[orig][des][i])
                          for (orig, des, i) in od_x_route), name='route_choice_approx')
        model.addConstrs((zeta[orig, des, i, l] <= p[orig, des, i]
                          for (orig, des, i) in od_x_route for l in destination[orig][des][i]),
                         name='route_choice_approx_p')
        model.addConstrs((zeta[orig, des, i, l] <= y[l]
                          for (orig, des, i) in od_x_route for l in destination[orig][des][i]),
                         name='route_choice_approx_y')
        model.addConstr(e.sum()/len(od_pairs) + lambd * gp.quicksum(a[i] for i in D[1:])/(dim-1)
                        <= L_bar, name='train_loss')
        model.addConstrs((e[orig, des] >=
                          gp.quicksum(phi[orig, des, i] - p[orig, des, i] * v_bar[orig][des][i] for i in route_id) -
                          gp.quicksum(c_feature_dict[orig, des][d] * omega[d] for d in D)
                          for orig, des in od_pairs), name='error_pos')
        model.addConstrs((e[orig, des] >=
                          gp.quicksum(c_feature_dict[orig, des][d] * omega[d] for d in D) -
                          gp.quicksum(phi[orig, des, i] - p[orig, des, i] * v_bar[orig][des][i] for i in route_id)
                          for orig, des in od_pairs), name='error_neg')
        model.addConstrs((a[d] >= omega[d] for d in D), name='reg_pos')
        model.addConstrs((a[d] >= -omega[d] for d in D), name='reg_pos')
        # add objective
        param_weight = self._gen_param_weight(dim=dim, pop=pop, o_feature_dict=o_feature_dict)
        obj = gp.quicksum(pop[des] * (phi[orig, des, i] - p[orig, des, i] * v_bar[orig][des][i])
                          for (orig, des, i) in od_x_route) + \
              omega.prod(param_weight)
        model.setObjective(obj, gp.GRB.MAXIMIZE)
        model.update()
        # store variables
        model._z = z
        model._omega = omega
        model._e = e
        return model

    @staticmethod
    def _gen_param_weight(dim, pop, o_feature_dict):
        vec = np.zeros(dim)
        for orig, des in o_feature_dict:
            vec += pop[des] * o_feature_dict[orig, des]
        return {idx: val for idx, val in enumerate(vec)}

    @staticmethod
    def _get_solution(model):
        z_val = model.getAttr('x', model._z)
        new_projects = [i for i in model._z if z_val[i] >= 1 - 1e-5]
        return new_projects

    @staticmethod
    def _get_od_utility(model, od_pairs, destination, pop, segs, beta, v_bar):
        zeta_val = model.getAttr('x', model._zeta)
        p_val = model.getAttr('x', model._p)
        phi_val = model.getAttr('x', model._phi)
        for orig, des in od_pairs:
            val = 0
            for idx, segments in enumerate(destination[orig][des]):
                val += phi_val[orig, des, idx]
                val -= p_val[orig, des, idx] * v_bar[orig][des][idx]
            #     print('prob', p_val[orig, des, idx], 'utility', phi_val[orig, des, idx] - v_bar[orig][des][idx])
            #     print(zeta_val.select(orig, des, idx, '*'))
            # print('------------------')
            # print(orig, des, val)
            # break

