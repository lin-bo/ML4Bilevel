import gurobipy as gp
import time
from solver.continuous.abstract_solver import AbstractSolver
from utils.check import file_existence
import numpy as np


class MipSolver(AbstractSolver):

    def solve(self, args, budget_proj, budget_sig, beta_1, time_limit=None, regenerate=False):
        print('\nreading the problem ...')
        od_pairs, destination, pop, G, n_nodes, projs, proj_costs, sig_costs, travel_time, edge2proj, T, M = self._args2params(args)
        beta = self._gen_betas(beta_1, T, M)
        print('The instance has {} od-pairs, {} projects, and {} nodes'.format(
            len(od_pairs), len(projs), n_nodes))
        print('compiling ...')
        tick = time.time()
        mip = self._construct_mip(od_pairs=od_pairs, pop=pop, budget_proj=budget_proj, budget_sig=budget_sig, regenerate=regenerate,
                                  proj_costs=proj_costs, sig_costs=sig_costs, beta=beta, G=G, T=T, M=M,
                                  travel_time=travel_time, projs=projs, edge2proj=edge2proj, time_limit=time_limit)
        print('  elapsed: {:.2f} sec'.format(time.time() - tick))
        print('solving ...')
        tick = time.time()
        mip.optimize()
        print('  obj val: {:.2f}'.format(mip.objVal))
        print('  elapsed: {:.2f} sec'.format(time.time() - tick))

    def _construct_mip(self, od_pairs, pop, budget_proj, budget_sig, beta, proj_costs, sig_costs, G, T, M, projs, edge2proj, travel_time, time_limit, regenerate):
        dir_name = './prob/{}/models/continuous/mip.mps'.format(self.ins_name)
        if file_existence(dir_name) and not regenerate:
            # gp.setParam('outputFlag', 0)
            model = gp.read(dir_name)
            # gp.setParam('outputFlag', 1)
            model = self._set_mip_params(model, time_limit)
            model = self._update_mip_rhs(model, budget_proj, budget_sig)
        else:
            # initialize the master problem
            model = gp.Model('master')
            model = self._set_mip_params(model, time_limit)
            # set parameters
            od_pops = {(orig, des): pop[des] for orig, des in od_pairs}
            od_edge_cost = {(orig, des, i, j): pop[des] * travel_time[i, j] for orig, des in od_pairs for (i, j) in G.edges()}
            projects = list(proj_costs.keys())
            signals = list(sig_costs.keys())
            # add variables
            z = model.addVars(od_pairs, name='z', vtype=gp.GRB.BINARY)
            u = model.addVars(od_pairs, name='u', vtype=gp.GRB.CONTINUOUS)
            y = model.addVars(projects, name='y', vtype=gp.GRB.BINARY)
            s = model.addVars(signals, name='s', vtype=gp.GRB.BINARY)
            x = model.addVars(od_pairs, G.edges(), name='x', vtype=gp.GRB.CONTINUOUS)
            # flow conservation - origin
            model.addConstrs(((gp.quicksum(x[orig, des, i, j] for i, j in G.in_edges(orig)) -
                               gp.quicksum(x[orig, des, i, j] for i, j in G.out_edges(orig)) + z[orig, des]
                               == 1) for orig, des in od_pairs), name='flow_balance_origin')
            # flow conservation - destination
            model.addConstrs(((gp.quicksum(x[orig, des, i, j] for i, j in G.in_edges(des)) -
                               gp.quicksum(x[orig, des, i, j] for i, j in G.out_edges(des)) - z[orig, des]
                               == -1) for orig, des in od_pairs), name='flow_balance_destination')
            # flow conservation - transmission
            model.addConstrs(((gp.quicksum(x[orig, des, g, h] for g, h in G.in_edges(i)) -
                               gp.quicksum(x[orig, des, g, h] for g, h in G.out_edges(i)) == 0)
                              for (orig, des) in od_pairs for i in G.nodes() if i != orig and i != des),
                             name='flow_balance_transshipment')
            # travel time limit
            model.addConstrs(((gp.quicksum(travel_time[i, j] * x[orig, des, i, j] for i, j in G.edges()) + M * z[orig, des]
                               <= M) for orig, des in od_pairs), name='travel_time_limit')
            # Travel time that exceeds T
            model.addConstrs(((gp.quicksum(travel_time[i, j] * x[orig, des, i, j] for i, j in G.edges()) - T
                               <= u[orig, des]) for orig, des in od_pairs), name='travel_time_exceeds_T')
            # edge design-forcing constraints
            model.addConstrs((x[orig, des, i, j] <= y[idx]
                              for idx in projects for i, j in projs[idx] for orig, des in od_pairs), name='edge_design')
            # node design-forcing constraints
            model.addConstrs((x[orig, des, i, j] <= s[i] + y[edge2proj[g, h]]
                              for i in signals for _, j in G.out_edges(i) for g, h in G.edges(i) for orig, des in od_pairs
                             if ((g, h) in edge2proj) and ((i != g) or (j != h))), name='node_design')
            # budget constraints
            model.addConstr(y.prod(proj_costs) <= budget_proj, name='project_budget')
            model.addConstr(s.prod(sig_costs) <= budget_sig, name='signal_budget')
            # set objective
            obj = beta[0] * z.prod(od_pops) + beta[1] * x.prod(od_edge_cost) + (beta[2] - beta[1]) * u.prod(od_pops)
            model.setObjective(obj, gp.GRB.MINIMIZE)
            if self.save_model:
                model.write(dir_name)
        return model

    @staticmethod
    def _set_mip_params(model, time_limit):
        # model.Params.outputFlag = 0
        if time_limit:
            model.Params.timeLimit = time_limit
        return model

    @staticmethod
    def _update_mip_rhs(model, budget_proj, budget_sig):
        model.setAttr("RHS", model.getConstrByName('project_budget'), budget_proj)
        model.setAttr("RHS", model.getConstrByName('signal_budget'), budget_sig)
        return model

