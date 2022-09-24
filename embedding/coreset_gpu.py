import torch
import numpy as np
import pandas as pd
import time
from scipy.spatial import distance_matrix


class PMedianHeuristicSolverGPU:
    def __init__(self, n_init, n_swap, n_term, quiet, feature, distance_metric='euc', device='mps'):
        self.n_init = n_init
        self.n_swap = n_swap
        self.n_term = n_term
        self.quiet = quiet
        self.distance_metric = distance_metric
        self.feature = feature
        self.nn = 0
        n, d = self.feature.shape
        self.nodes = np.array(list(range(n)))
        self.device = torch.device(device)

    def solve(self, p, kappa, idx):
        n, d = self.feature.shape
        obj_best = 1e8
        selected_best = []
        records = []
        for _ in range(self.n_init):
            tick = time.time()
            selected = self._initialize(n, p)
            obj = self._obj_eval(selected, kappa)
            inter_cnt = 0
            improve = True
            while inter_cnt <= self.n_term and improve:
                if not self.quiet:
                    print('\n**********************')
                    print('iteration {}'.format(inter_cnt))
                inter_cnt += 1
                # alternate phase
                selected, obj = self._alternate(selected, kappa, obj)
                if not self.quiet:
                    print('alter: {}'.format(obj))
                # swap phase
                obj_archive = obj
                params = []
                for i in range(self.n_swap):
                    params.append((selected, kappa, obj))
                    selected, obj = self._swap(selected, kappa, obj)
                improve = True if obj < obj_archive else False
                if not self.quiet:
                    print('swap: {}'.format(obj))
            if obj < obj_best:
                obj_best = obj
                selected_best = selected.copy()
                records.append([obj_best, selected_best.copy()])
            df = pd.DataFrame(records, columns=['obj', 'medians'])
            # df.to_csv('/content/res_{}_ratio_20.csv'.format(idx), index=False)
            df.to_csv('./prob/trt/pmedian/pmedian_gpu/sample{}_p{}.csv'.format(idx, p), index=False)
            print("obj: {:.4f}, time: {:.4f}".format(obj_best, time.time() - tick))
        return selected_best, obj_best

    @staticmethod
    def _initialize(n, p):
        return np.random.choice(list(range(n)), p, replace=False).tolist()

    def _obj_eval(self, selected, kappa):
        dist_mat = pairwise_distance(self.feature, self.feature[selected, :], self.distance_metric)
        dist_mat, _ = torch.min(dist_mat, dim=1, keepdim=True)
        pmedian_obj = dist_mat[:, :kappa].sum()
        return pmedian_obj

    def _swap(self, selected, kappa, obj):
        n, d = self.feature.shape
        # generate random nodes inside and outside the selected set S'
        s = np.random.choice(selected, 1)[0]
        k = np.random.choice(range(n), 1)[0]
        while k in selected:
            k = np.random.choice(range(n), 1)[0]
        # generate the new selected set S''
        selected_new = selected.copy()
        selected_new.remove(s)
        selected_new.append(k)
        # evaluation and comparison
        obj_new = self._obj_eval(selected_new, kappa)
        if obj > obj_new:
            obj = obj_new
            selected = selected_new
        # print('swap', obj, obj_new)
        return selected, obj

    def _alternate(self, selected, kappa, obj):
        improve = True
        # assignment
        while improve:
            improve = False
            dist_mat = pairwise_distance(self.feature, self.feature[selected, :], self.distance_metric)
            self.nn = torch.argmin(dist_mat, dim=1).to(torch.device('cpu')).reshape((-1, 1))
            # solving one medians
            selected_new = []
            params = []
            for idx, s in enumerate(selected):
                new_median = self._one_median(idx)
                selected_new.append(new_median)
            obj_new = self._obj_eval(selected_new, kappa)
            if obj_new < obj:
                selected = selected_new.copy()
                obj = obj_new
                improve = True
            print("alter", obj, obj_new)
        return selected, obj

    def _one_median(self, idx):
        flag = (self.nn == idx).sum(axis=1).to(torch.bool)
        assigned_nodes = self.nodes[flag]
        if len(assigned_nodes) > 10000:
            assigned_nodes = np.random.choice(assigned_nodes, 10000, replace=False)
        points = self.feature[assigned_nodes]
        tot_dist = pairwise_distance(points, points).sum(dim=1)
        new_median = torch.argmin(tot_dist).to(torch.device('cpu')).item()
        return assigned_nodes[new_median]

    @staticmethod
    def _get_best_swap(res):
        best_selected, best_obj = [], 1e8
        for selected, obj in res:
            if obj < best_obj:
                best_obj = obj
                best_selected = selected.copy()
        return best_selected, best_obj


def pairwise_distance(points_1, points_2, metric='cosine'):
    if metric == 'euc':
        return distance_matrix(points_1, points_2)
    elif metric == 'cosine':
        dist_mat = torch.matmul(points_1, points_2.T)
        dist_mat = 1 - dist_mat
        return dist_mat
    else:
        raise ValueError('metric {} is not defined'.format(metric))
