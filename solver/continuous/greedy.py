from utils.functions import cal_acc, dump_file
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path_length
from tqdm import tqdm
import pandas as pd
import multiprocessing


class GreedySolver:

    def __init__(self, metric='abs'):
        self.metric = metric
        self.n_worker = 8

    def search(self, projs, args, selected, curr_acc, remained):
        best_val, best_inc, best_idx, new_curr = 0, 0, -1, curr_acc
        proj_costs = args['project_costs']
        for p in tqdm(projs):
            if proj_costs[p] > remained:
                continue
            acc = cal_acc(args, selected + [p], [], impedence='time')
            val = acc - curr_acc if self.metric == 'abs' else (acc - curr_acc) / proj_costs[p]
            if val > best_val:
                best_val = val
                new_curr = acc
                best_idx = p
        return best_val, new_curr, best_idx

    def solve(self, args, budget, region):
        # extract info
        projs = list(range(len(args['projects'])))
        proj_costs = args['project_costs']
        projs = [p for p in projs if proj_costs[p]]
        # intialize parameters
        allocated = 0
        curr_acc = cal_acc(args, [], [], impedence='time')
        idx = 0
        selected = []
        records = []
        print('initial acc: {}'.format(curr_acc))
        while allocated < budget:
            remained = budget - allocated
            print('\nround {}'.format(idx))
            best_val, curr_acc, best_idx = self.search(projs, args, selected, curr_acc, remained)
            if best_idx == -1:
                break
            projs.remove(best_idx)
            selected.append(best_idx)
            allocated += proj_costs[best_idx]
            idx += 1
            records.append((allocated, curr_acc, selected.copy()))
            df = pd.DataFrame(records, columns=['allocated', 'acc', 'selected'])
            dump_file('./prob/trt/res/greedy_{}_{}_{}.pkl'.format(self.metric, budget, region), df)
            print('selected: {}, new acc: {}, metric: {}, allocated: {}'.format(best_idx, curr_acc, best_val, allocated))


class GreedySolverPar:

    def __init__(self, metric='abs', n_workers=8, potential='populations'):
        self.metric = metric
        self.n_workers = n_workers
        self.potential = potential

    def search(self, projs, args, selected, curr_acc, remained):
        proj_costs = args['project_costs']
        params = [(args, selected + [p], [], p, 'time', self.potential) for p in projs]
        pool = multiprocessing.Pool(self.n_workers)
        data = list(tqdm(pool.imap(self._cal_acc, params), total=len(proj_costs)))
        pool.close()
        return self._select_best(data, curr_acc, remained, proj_costs)

    def solve(self, args, budget):
        # extract info
        projs = list(range(len(args['projects'])))
        proj_costs = args['project_costs']
        projs = [p for p in projs if proj_costs[p]]
        # intialize parameters
        allocated = 0
        curr_acc = cal_acc(args, [], [], impedence='time')
        idx = 0
        selected = []
        records = []
        print('initial acc: {}'.format(curr_acc))
        while allocated < budget:
            remained = budget - allocated
            print('\nround {}'.format(idx))
            best_val, curr_acc, best_idx = self.search(projs, args, selected, curr_acc, remained)
            if best_idx == -1:
                break
            projs.remove(best_idx)
            selected.append(best_idx)
            allocated += proj_costs[best_idx]
            idx += 1
            records.append((allocated, curr_acc, selected.copy()))
            df = pd.DataFrame(records, columns=['allocated', 'acc', 'selected'])
            dump_file('./prob/trt/res/greedy_{}_{}_par.pkl'.format(self.metric, self.potential), df)
            print('selected: {}, new acc: {}, metric: {}, allocated: {}'.format(best_idx, curr_acc, best_val, allocated))

    @staticmethod
    def _cal_acc(inputs):
        # retrieve information
        args, new_projects, new_signals, added_proj, impedance, potential = inputs
        G_curr = args['G_curr'].copy()
        G = args['G'].copy()
        T = args['travel_time_limit']
        pop = args[potential]
        destinations = args['destinations']
        projs = args['projects']
        travel_time = args['travel_time']
        unsig_inters = args['signal_costs']
        # get new edges
        new_edges, new_nodes = [], set([])
        for idx in new_projects:
            new_edges += projs[idx]
            for (i, j) in projs[idx]:
                new_nodes.add(i)
                new_nodes.add(j)
        new_nodes = [idx for idx in new_nodes if idx in unsig_inters and idx not in new_signals]
        for idx in new_signals + new_nodes:
            new_edges += [(i, j) for (i, j) in G.out_edges(idx) if j in destinations]
        # get attributes for new edges
        edges_w_attr = [(i, j, {impedance: travel_time[i, j]}) for (i, j) in new_edges]
        # add new edges
        G_curr.add_edges_from(edges_w_attr)
        acc = 0
        for orig in destinations:
            lengths = single_source_dijkstra_path_length(G=G_curr, source=orig, cutoff=T, weight=impedance)
            reachable_des = [des for des in lengths if des in destinations[orig]]
            for des in reachable_des:
                acc += pop[des]
        return acc, added_proj

    def _select_best(self, data, curr_acc, remained, proj_costs):
        best_val, best_inc, best_idx, new_curr = 0, 0, -1, curr_acc
        for acc, p in data:
            if proj_costs[p] > remained:
                continue
            val = acc - curr_acc if self.metric == 'abs' else (acc - curr_acc) / proj_costs[p]
            if val > best_val:
                best_val = val
                new_curr = acc
                best_idx = p
        return best_val, new_curr, best_idx
