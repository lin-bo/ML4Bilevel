import numpy as np
from utils.functions import one2all_dist, des2od, cal_od_utilies, pairwise_distance
from scipy.spatial import distance_matrix
import gurobipy as gp
import time
from tqdm import tqdm
from utils.functions import load_file


class PMedianSolver:

    def __init__(self):
        pass

    def solve(self, points, p, W=[], return_obj=False):
        """
        solve the given k center instance exactly
        :param points: (n x d) array
        :param p: the number of center we want to pick
        :return:
        """
        # calculate cost
        if len(W) > 0:
            C = W
        else:
            C = distance_matrix(points, points)
        self.model = self._construct_mip(C=C, p=p)
        self.model.optimize()
        self.pair_idx = self._get_centers(self.model)
        self.cnt = 0
        if return_obj:
            return self.pair_idx, self.model.objVal
        return self.pair_idx

    def purturb(self, frac=0.5, seed=None):
        if self.cnt == 0:
            self.cnt += 1
            return self.pair_idx
        if seed:
            np.random.seed(seed)
        n = np.max([int(len(self.pair_idx) * frac), 2])
        fixed, varied = self._gen_fix_vary(self.pair_idx, n)
        self.model = self._add_perturb_constraints(self.model, fixed, varied)
        self.model.optimize()
        new_pair_idx = self._get_centers(self.model)
        self.model = self._remove_perturb_constraints(self.model, fixed, varied)
        return new_pair_idx

    @staticmethod
    def _construct_mip(C, p):
        # set parameters
        n = C.shape[0]
        N = list(range(n))
        obj_weight = {(i, j): C[i, j] for i in N for j in N}
        # intitialize model
        model = gp.Model('pcenter')
        model.Params.outputFlag = 0
        # add variables
        x = model.addVars(N, name='x', vtype=gp.GRB.BINARY)
        y = model.addVars(N, N, name='y', vtype=gp.GRB.BINARY)
        # add constraints
        model.addConstrs((gp.quicksum(y[i, j] for j in N) == 1 for i in N), name='node_assignment')
        model.addConstrs((y[i, j] <= x[j] for i in N for j in N), name = 'no_selection_no_assignment')
        model.addConstr(x.sum() == p, name='node_selection')
        # set objective
        model.setObjective(y.prod(obj_weight), gp.GRB.MINIMIZE)
        # store variables
        model._x = x
        return model

    @staticmethod
    def _add_perturb_constraints(model, fixed, varied):
        # add constraints
        model.addConstrs((model._x[i] == 1 for i in fixed), name='fix')
        model.addConstrs((model._x[i] == 0 for i in varied), name='vary')
        return model

    @staticmethod
    def _remove_perturb_constraints(model, fixed, varied):
        for i in fixed:
            model.remove(model.getConstrByName('fix[{}]'.format(i)))
        for i in varied:
            model.remove(model.getConstrByName('vary[{}]'.format(i)))
        model.update()
        return model

    @staticmethod
    def _gen_fix_vary(pair_idx, n):
        varied_idx = np.random.choice(pair_idx, n, replace=False)
        fixed_idx = np.array([idx for idx in pair_idx if idx not in varied_idx])
        return fixed_idx, varied_idx

    @staticmethod
    def _get_centers(model):
        x_val = model.getAttr('x', model._x)
        return [idx for idx, val in x_val.items() if val > 1-1e-10]


class PMedianHeuristicSolver:

    def __init__(self, n_init, n_swap, n_term, quiet, feature, uneven_penalty=0.1,
                 distance_metric='cosine', n_workers=8, random_seed=None, weight=[]):
        self.n_init = n_init
        self.n_swap = n_swap
        self.n_term = n_term
        self.quiet = quiet
        self.distance_metric = distance_metric
        self.feature = feature
        self.nn = 0
        n, d = self.feature.shape
        self.nodes = np.array(list(range(n)))
        self.n_workers = n_workers
        self.uneven_penalty = uneven_penalty
        self.weight = np.array(weight).reshape((n, 1)) if len(weight) > 0 else np.ones((n, 1))
        if random_seed:
            np.random.seed(random_seed)

    def solve(self, p, kappa):
        n, d = self.feature.shape
        obj_best = 1e8
        selected_best = []
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
                # multiprocessing
                # pool = multiprocessing.Pool(self.n_workers)
                # res_swap = pool.starmap(self._swap, params)
                # pool.close()
                # selected, obj = self._get_best_swap(res_swap)
                improve = True if obj < obj_archive else False
                if not self.quiet:
                    print('swap: {}'.format(obj))
            if obj < obj_best:
                obj_best = obj
                selected_best = selected.copy()
            # print("obj: {:.4f}, time: {:.4f}".format(obj_best, time.time() - tick))
        return selected_best, obj_best

    @staticmethod
    def _initialize(n, p):
        return np.random.choice(list(range(n)), p, replace=False).tolist()

    def _obj_eval(self, selected, kappa):
        dist_mat = pairwise_distance(self.feature, self.feature[selected, :], self.distance_metric)
        dist_mat = np.delete(dist_mat, selected, axis=0)
        weight = np.delete(self.weight.copy(), selected, axis=0)
        # dist_mat = np.sort(dist_mat, axis=1)
        pmedian_obj = (np.sort(dist_mat, axis=1)[:, :kappa] * weight).sum()
        neighbors = np.argsort(dist_mat, axis=1)[:, :kappa].reshape(-1)
        _, cnts = np.unique(neighbors, return_counts=True)
        avg_cnt = len(dist_mat) / len(cnts)
        even_assign_obj = np.abs(cnts - avg_cnt).sum() * 0.2  # just to scale the two objectives a bit
        return pmedian_obj * (1 - self.uneven_penalty) + even_assign_obj * self.uneven_penalty

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
        return selected, obj

    def _alternate(self, selected, kappa, obj):
        improve = True
        # assignment
        while improve:
            improve = False
            dist_mat = pairwise_distance(self.feature, self.feature[selected], self.distance_metric)
            self.nn = np.argsort(dist_mat, axis=1)[:, :kappa]
            # solving one medians
            selected_new = []
            params = []
            for idx, s in enumerate(selected):
                params.append((idx, ))
                new_median = self._one_median(idx)
                selected_new.append(new_median)
            obj_new = self._obj_eval(selected_new, kappa)
            if obj_new < obj:
                selected = selected_new.copy()
                obj = obj_new
                improve = True
        return selected, obj

    def _one_median(self, idx):
        flag = (self.nn == idx).sum(axis=1).astype(bool)
        assigned_nodes = self.nodes[flag]
        points = self.feature[assigned_nodes]
        dist_mat = pairwise_distance(points, points, self.distance_metric)
        tot_dist = dist_mat.sum(axis=1)
        return assigned_nodes[np.argmin(tot_dist)]

    @staticmethod
    def _get_best_swap(res):
        best_selected, best_obj = [], 1e8
        for selected, obj in res:
            if obj < best_obj:
                best_obj = obj
                best_selected = selected.copy()
        return best_selected, best_obj


def greedy_kcenter(feature, n, k, repeat=30, tol=0.1, random_seed=None, return_delta=False):
    """
    select a core set from the given nodes
    :param feature: n_pairs x d array,
    :param n: the size of the core set
    :param k: the number of nearest neighbors we want to consider
    :param repeat: the number of times we want to repeat the search
    :param tol: tolerance (for more stable result)
    :param random_seed: random seed
    :return: a list of selected core set nodes, the KNN in the core set for each od pair
    """
    # set random seed
    if random_seed:
        np.random.seed(random_seed)
    # set parameters
    n_pairs = len(feature)
    n_outliers = np.max([1, int(n_pairs * tol)])
    # print('selecting {} from {} allowing {} outliers'.format(n, n_pairs, n_outliers))
    seeds = np.random.choice(np.arange(0, n_pairs), repeat, replace=False)
    best_delta = 1e12
    best_coreset = []
    best_dist_mat = np.zeros((1, n_pairs))
    # start searching
    for idx in range(repeat):
        coreset = []
        dist_mat = np.zeros((1, n_pairs))
        cnt = 0
        # randomly pick an initial pair
        while cnt < n:
            node = seeds[idx] if cnt == 0 else \
                np.argsort(dist_mat[1:].min(axis=0))[-n_outliers]
            dist_mat = np.concatenate([dist_mat, one2all_dist(feature, feature[node])], axis=0)
            coreset.append(node)
            cnt += 1
        delta = np.sort(dist_mat[1:].min(axis=0))[-n_outliers]
        if delta < best_delta:
            best_delta = delta
            best_coreset = coreset.copy()
            best_dist_mat = dist_mat[1:].copy()
    neighbor_indices = np.argsort(best_dist_mat, axis=0)[:k].T
    neighbors = np.take(best_coreset, neighbor_indices)
    if return_delta:
        return best_coreset, neighbors, best_delta
    else:
        return best_coreset, neighbors


def iter_kcenter(feature, n, k, repeat=30, tol=0.1, random_seed=None):
    # calculate the full distance matrix
    dist_mat = distance_matrix(feature, feature)
    # find a starting coreset greedily
    coreset, _, delta = greedy_kcenter(feature, n, k, repeat, 0, random_seed, return_delta=True)
    # set the upper and lower bounds
    ub, lb = delta, delta / 2
    while ub >= lb + 1e-3:
        pass


class IterKCenterFeasibilityCheck:
    def __init__(self, feature, k):
        n, _ = feature.shape
        self.model = self._build_model(n, k)

    def _build_model(self, n, k):
        # initialize
        model = gp.Model('check')
        # add variables
        s = model.addVars(n, name='s', vtype=gp.GRB.BINARY)
        a = model.addVars(n, n, name='a', vtype=gp.GRB.BINARY)
        # add constraints
        model.addConstr(s.sum() == k, name='budget')
        model.addConstrs((a.select(i, '*').sum() == 1
                          for i in range(n)), name='unique_assignment')
        model.addConstrs((a[i, j] <= s[j] for i in range(n) for j in range(n)), name='assign_con_select')
        # set obj
        model.setObjective(0, gp.GRB.MINIMIZE)
        # store variables
        model._s = s
        model._a = a
        return model

    def check(self):
        pass


def check_center_obj(feature, coreset, tol=0.1):
    """
    given a solution to the k-center problem, evaluate its objective value (and the obj val considering tolerance)
    :param feature: (n x d) array
    :param coreset: k-dimensional list
    :param tol: float between 0 and 1
    :return: obj vals without and with the tolerance consideration
    """
    # set parameters
    n, d = feature.shape
    n_outliers = np.max([1, int(n * tol)])
    dist_mat = np.ones((1, n)) * 100
    # calculate distance matrix (centers to others)
    for c in coreset:
        dists = one2all_dist(feature, feature[c])
        dist_mat = np.concatenate([dist_mat, dists], axis=0)
    dist_mat = dist_mat[1:, :]
    # w/o tolerance
    dist_min = dist_mat.min(axis=0)
    delta_wo = dist_min.max()
    # avg distance
    dist_avg = dist_min.mean()
    # w/ tolerance
    delta_w = np.sort(dist_min)[-n_outliers]
    return delta_wo, delta_w, dist_avg


def gen_argument(args, coreset, neighbors, equity=False, alpha=1, feature=[], potential='populations'):
    """
    generate new problem argument based on the selected coreset
    :param args: instance argument for the original problem
    :param coreset: a list of indices of the selected od pairs (for orig for des in destination[orig])
    :param neighbors: 2-d array, each row is the indices of the three most similar od pairs
    :return: a new argument dict
    """
    # load information
    destinations = args['destinations']
    pop = args[potential]
    od_pairs = des2od(destinations)
    node2score, _ = load_file('./data/on_marg_index/node2score.pkl')
    # set new od pairs
    new_pairs = [od_pairs[c] for c in coreset]
    # set new destinations
    destinations_new = {k: {} for k in list(destinations.keys())}
    for orig, des in new_pairs:
        destinations_new[orig][des] = destinations[orig][des]
    # update arguments
    args_new = args.copy()
    args_new['destinations'] = destinations_new
    args_new['neighbors'] = neighbors
    args_new['od_pairs_init'] = od_pairs
    # calculate weights
    if len(neighbors) > 0:
        weights = {(orig, des): 0 for orig, des in new_pairs}
        for i, neighbor in enumerate(neighbors):
            if od_pairs[i] in weights:
                weights[od_pairs[i]] += pop[od_pairs[i][1]]
            else:
                for j in neighbor:
                    marg_score = 1 if not equity else (node2score[od_pairs[j][0]] * 10) ** alpha
                    weights[od_pairs[j]] += pop[od_pairs[i][1]] * marg_score / len(neighbor)
        args_new['weights'] = weights
    # add feature info
    if len(feature) > 0:
        in_sample_vecs = feature[coreset]
        flag = np.ones(feature.shape[0])
        flag[coreset] = 0
        flag = flag.astype(bool)
        out_of_sample_pop = np.array([pop[des] for (orig, des) in np.array(od_pairs)[flag]]).reshape((-1, 1))
        out_of_sample_vec = (feature[flag] * out_of_sample_pop).sum(axis=0)
        args_new['in_sample'] = {od_pairs[c]: {i: val for i, val in enumerate(in_sample_vecs[idx])}
                                 for idx, c in enumerate(coreset)}
        args_new['out_of_sample'] = {idx: out_of_sample_vec[idx] for idx in range(feature.shape[1])}
    return args_new


def gen_argument_prod(args, coreset, neighbors, std_dict, alpha=1.02):
    """
    generate new problem argument based on the selected coreset
    :param args: instance argument for the original problem
    :param coreset: a list of indices of the selected od pairs (for orig for des in destination[orig])
    :param neighbors: 2-d array, each row is the indices of the three most similar od pairs
    :return: a new argument dict
    """
    # load information
    destinations = args['destinations']
    pop = args['populations']
    od_pairs = des2od(destinations)
    init_utility = cal_od_utilies(args=args, new_projects=[], alpha=alpha)
    # set new od pairs
    new_pairs = [od_pairs[c] for c in coreset]
    # set new destinations
    destinations_new = {k: {} for k in list(destinations.keys())}
    for orig, des in new_pairs:
        destinations_new[orig][des] = destinations[orig][des]
    # update arguments
    args_new = args.copy()
    args_new['destinations'] = destinations_new
    # calculate weights
    if len(neighbors) > 0:
        weights = {(orig, des): 0 for orig, des in new_pairs}
        for i, neighbor in enumerate(neighbors):
            if od_pairs[i] in weights:
                weights[od_pairs[i]] += pop[od_pairs[i][1]]
            else:
                for j in neighbor:
                    weights[od_pairs[j]] += pop[od_pairs[i][1]] * init_utility[od_pairs[i]] * std_dict[od_pairs[i]] \
                                            / init_utility[od_pairs[j]] / std_dict[od_pairs[j]] / len(neighbor)
        args_new['weights'] = weights
    return args_new


def find_neighbors(feature, coreset, k):
    """
    find each point's k nearest neighbors in the coreset
    :param feature: n x d array
    :param k: number of neighbors for each point
    :return: n x k array, each row contains the indices of the k-nearest neighbors
    """
    n, d = feature.shape
    dist_mat = np.zeros((1, n))
    # for c in coreset:
    #     dists = one2all_dist(feature, feature[c])
    #     dist_mat = np.concatenate([dist_mat, dists], axis=0)
    if len(coreset) <= 1000:
        dist_mat = pairwise_distance(feature[coreset], feature, 'cosine')
        neighbor_indices = np.argsort(dist_mat, axis=0)[:k].T
    else:
        n_chunk = int(len(coreset) // 500)
        s_chunk = int(len(feature) // n_chunk) + int(len(feature) % n_chunk > 0)
        # chunks = [(idx * s_chunk, np.min([(idx + 1) * s_chunk, len(feature)])) for idx in range(n_chunk)]
        # nn_idx = [np.argsort(pairwise_distance(feature[coreset], feature[cs: ce], 'cosine'), axis=0)[:k].T for cs, ce in chunks]
        nn_idx = []
        for idx in tqdm(range(n_chunk)):
            cs, ce = idx * s_chunk, np.min([(idx + 1) * s_chunk, len(feature)])
            nn_idx.append(np.argsort(pairwise_distance(feature[coreset], feature[cs: ce], 'cosine'), axis=0)[:k].T)
        neighbor_indices = np.concatenate(nn_idx, axis=0)
    neighbors = np.take(coreset, neighbor_indices)
    return neighbors


def gen_feature_dict(feature, coreset, od_pairs):
    coreset_feature = feature[coreset]
    flag = np.ones(feature.shape[0], dtype=bool)
    flag[coreset] = False
    other_feature = feature[flag]
    c_feature_dict, o_feature_dict = {}, {}
    for idx in range(len(coreset)):
        c_feature_dict[od_pairs[coreset[idx]]] = coreset_feature[idx]
    cnt = 0
    for orig, des in od_pairs:
        if (orig, des) not in c_feature_dict:
            o_feature_dict[orig, des] = other_feature[cnt]
            cnt += 1
    return c_feature_dict, o_feature_dict
