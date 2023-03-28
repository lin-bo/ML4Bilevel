import numpy as np
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path_length
import pickle
from tqdm import tqdm
import os
from gensim.models import Word2Vec
from utils.functions import des2od, one2all_dist, cal_od_utility, dump_file, load_file
from sklearn.preprocessing import StandardScaler
import multiprocessing


class ScenarioSampler:

    def __init__(self, P, U, save=False, multiproc=False, n_workers=4, weight='travel_time'):
        self.P = P
        self.U = U
        self.save = save
        self.multiproc = multiproc
        self.n_workers = n_workers
        self.weight = weight

    def sample_train(self, args, n, ins_name):
        conn_mat_path = './prob/{}/emb/time_matrix/connection_matrix-p{}-u{}-n{}.pkl'.format(ins_name, self.P, self.U, n)
        time_mat_path = './prob/{}/emb/time_matrix/time_matrix-p{}-u{}-n{}.pkl'.format(ins_name, self.P, self.U, n)
        conn_matrix, time_matrix = self._sample(args, n, conn_mat_path, time_mat_path)
        # save the matrices
        if self.save:
            # with open(conn_mat_path, 'wb') as f:
            #     pickle.dump(np.array(conn_matrix), f)
            with open(time_mat_path, 'wb') as f:
                pickle.dump(np.array(time_matrix), f)
        return conn_matrix, time_matrix

    def sample_test(self, args, n, ins_name):
        conn_mat_path = './prob/{}/emb/connection_matrix-p{}-u{}-n{}_test.pkl'.format(ins_name, self.P, self.U, n)
        time_mat_path = './prob/{}/emb/time_matrix-p{}-u{}-n{}_test.pkl'.format(ins_name, self.P, self.U, n)
        conn_matrix, time_matrix = self._sample(args, n, conn_mat_path, time_mat_path)
        # save the matrices
        if self.save:
            with open(conn_mat_path, 'wb') as f:
                pickle.dump(np.array(conn_matrix), f)
            with open(time_mat_path, 'wb') as f:
                pickle.dump(np.array(time_matrix), f)
        return conn_matrix, time_matrix

    def sample_by_chunk(self, args, ins_name, chunk_size, sidx, eidx):
        for c in range(sidx, eidx):
            conn_mat_path = './prob/{}/emb/time_matrix/connection_matrix-p{}-u{}-chunk{}.pkl'.format(ins_name, self.P, self.U, c)
            time_mat_path = './prob/{}/emb/time_matrix/time_matrix-p{}-u{}-chunk{}.pkl'.format(ins_name, self.P, self.U, c)
            conn_matrix, time_matrix = self._sample(args, chunk_size, conn_mat_path, time_mat_path)
            # save the matrices
            print(time_matrix.shape)
            if self.save:
                # with open(conn_mat_path, 'wb') as f:
                #     pickle.dump(np.array(conn_matrix), f)
                with open(time_mat_path, 'wb') as f:
                    pickle.dump(np.array(time_matrix), f)

    def _sample(self, args, n, conn_mat_path, time_mat_path):
        # check if the matrices have been generated or not
        if os.path.exists(conn_mat_path) and os.path.exists(time_mat_path):
            print('loading ...')
            with open(conn_mat_path, 'rb') as f:
                conn_matrix = pickle.load(f)
            with open(time_mat_path, 'rb') as f:
                time_matrix = pickle.load(f)
        else:
            # retrieve instance data
            projects = args['projects']
            signals = list(args['signal_costs'].keys())
            destinations = args['destinations']
            G_curr = args['G_curr'].copy()
            G = args['G']
            travel_time = args['travel_time']
            T = args['travel_time_limit']
            M = args['travel_time_max']
            # generate the number of projects/signals to select each time
            num_projs = np.random.choice(range(int(self.P/2), self.P), size=n, replace=True).reshape((-1, 1))
            if self.U > 1:
                num_sigs = np.random.choice(range(1, self.U), size=n, replace=True).reshape((-1, 1))
            else:
                num_sigs = np.zeros(n, dtype=int).reshape((-1, 1))
            config = np.concatenate([num_projs, num_sigs], axis=1)
            # generate random sub-graph
            conn_matrix, time_matrix = [], []
            print('sampling scenarios ...')
            for (n_proj, n_sig) in tqdm(config):
                new_projects, new_signals = self._gen_subgraph(range(len(projects)), signals, n_proj, n_sig)
                c_vec, t_vec = self._check_connectivity(G, G_curr.copy(), M, destinations, travel_time, projects, signals, new_projects, new_signals)
                conn_matrix.append(c_vec)
                time_matrix.append(t_vec)
        return np.array(conn_matrix), np.array(time_matrix)

    @staticmethod
    def _gen_subgraph(projects, signals, n_proj, n_sig):
        """
        randomly select n1 projects and n2 signals
        :param projects: a list of given projects
        :param signals: a list of given signals
        :param n1:
        :param n2:
        :return: a list of selected projects and a list of selected signals
        """
        projs = np.random.choice(projects, n_proj, replace=False)
        if n_sig > 0:
            sigs = np.random.choice(signals, n_sig, replace=False)
        else:
            sigs = np.array([])
        return projs, sigs

    def _check_connectivity(self, G, G_curr, T, destinations, travel_time, projects, signals, new_projects, new_signals):
        new_edges, new_nodes = [], set([])
        for idx in new_projects:
            new_edges += projects[idx]
            for (i, j) in projects[idx]:
                new_nodes.add(i)
                new_nodes.add(j)
        new_nodes = [idx for idx in new_nodes if idx in signals and idx not in new_signals]
        for idx in new_signals.tolist() + new_nodes:
            new_edges += [(i, j) for (i, j) in G.out_edges(idx) if j in destinations]
        # get attributes for new edges
        edges_w_attr = [(i, j, {self.weight: travel_time[i, j]}) for (i, j) in new_edges]
        # add new edges
        G_curr.add_edges_from(edges_w_attr)
        if not self.multiproc:
            c_vec, t_vec = [], []
            for orig in destinations:
                lengths = single_source_dijkstra_path_length(G=G_curr, source=orig, cutoff=T, weight=self.weight)
                c, t = [0 for _ in destinations[orig]], [-1 for _ in destinations[orig]]
                for idx, des in enumerate(destinations[orig]):
                    c[idx] = 1 if des in lengths else 0
                    t[idx] = lengths[des] if des in lengths else -1
                c_vec += c
                t_vec += t
        else:
            params = []
            self.G_curr = G_curr
            self.destinations = destinations
            for orig in destinations:
                params.append((orig, T))
            pool = multiprocessing.Pool(self.n_workers)
            data = pool.starmap(self._single_origin_conn, params)
            pool.close()
            c_vec, t_vec = self._data2vec(data, list(destinations.keys()))
        return c_vec, t_vec

    def _single_origin_conn(self, orig, T):
        lengths = single_source_dijkstra_path_length(G=self.G_curr, source=orig, cutoff=T, weight=self.weight)
        c, t = [0 for _ in self.destinations[orig]], [-1 for _ in self.destinations[orig]]
        for idx, des in enumerate(self.destinations[orig]):
            c[idx] = 1 if des in lengths else 0
            t[idx] = lengths[des] if des in lengths else -1
        return orig, c, t

    @staticmethod
    def _data2vec(data, origins):
        data = {orig: {'c': c, 't': t} for orig, c, t in data}
        c_vec, t_vec = [], []
        for orig in origins:
            c_vec += data[orig]['c']
            t_vec += data[orig]['t']
        return c_vec, t_vec


class ScenarioSamplerRoute:

    def __init__(self, P, save=False):
        self.P = P
        self.save = save

    def sample_train(self, args, n, ins_name):
        utility_mat_path = './prob/{}/emb/utility_matrix-p{}-n{}.pkl'.format(ins_name, self.P, n)
        utility_matrix = self._sample(args, n, utility_mat_path)
        # save the matrices
        if self.save:
            with open(utility_mat_path, 'wb') as f:
                pickle.dump(np.array(utility_matrix), f)
        return utility_matrix

    def sample_test(self, args, n, ins_name):
        utility_mat_path = './prob/{}/emb/utility_matrix-p{}-n{}_test.pkl'.format(ins_name, self.P, n)
        utility_matrix = self._sample(args, n, utility_mat_path)
        # save the matrices
        if self.save:
            with open(utility_mat_path, 'wb') as f:
                pickle.dump(np.array(utility_matrix), f)
        return utility_matrix

    def _sample(self, args, n, utility_mat_path, alpha=1.02):
        # check if the matrices have been generated or not
        if os.path.exists(utility_mat_path):
            print('loading ...')
            with open(utility_mat_path, 'rb') as f:
                utility_matrix = pickle.load(f)
        else:
            # retrieve instance data
            projects = args['projects']
            od_routes = args['od_routes']
            v_bar = args['v_bar']
            # generate the number of projects/signals to select each time
            num_projs = np.random.choice(range(1, self.P), size=n, replace=True).reshape((-1, 1))
            # generate random sub-graph
            utility_matrix = []
            print('sampling scenarios ...')
            for n_proj in tqdm(num_projs):
                new_projects = self._gen_subgraph(range(len(projects)), n_proj)
                u_vec = self._check_utility(od_routes, v_bar, new_projects, projects, alpha)
                utility_matrix.append(u_vec)
        return np.array(utility_matrix)

    @staticmethod
    def _gen_subgraph(projects, n_proj):
        """
        randomly select n1 projects and n2 signals
        :param projects: a list of given projects
        :param n1:
        :param n2:
        :return: a list of selected projects
        """
        projs = np.random.choice(projects, n_proj, replace=False)
        return projs

    @staticmethod
    def _check_utility(od_routes, v_bar, new_projects, projects, alpha):
        # set parameters
        n_orig = len(od_routes)
        # get edges along new project
        new_proj_edges = []
        for idx in new_projects:
            new_proj_edges += projects[idx]
        new_proj_edges = set(new_proj_edges)
        # calculate the utility
        u_vec = []
        for orig, destinations in od_routes.items():
            for des, routes in destinations.items():
                u = cal_od_utility(routes, v_bar[orig][des], new_proj_edges, n_orig, alpha)
                u_vec.append(u)
        return u_vec


class DeepWalk_variant:

    def __init__(self, random_seed=None, save=False, variant_type='exp'):
        if random_seed:
            np.random.seed(random_seed)
        self.save = save
        self.variant_type = variant_type

    def node2vec(self, ins_name, suffix, weights, walk_per_node=50, walk_length=10, dim=32):
        w2v_path = './prob/{}/emb/{}/emb_{}_dim{}'.format(ins_name, self.variant_type, suffix, dim)
        if os.path.exists(w2v_path):
            print('already trained, loading w2v model ...')
            model = Word2Vec.load(w2v_path)
        else:
            nodes = range(len(weights))
            print('number of paths to sample: {}'.format(len(nodes) * walk_per_node))
            print('sampling paths ...')
            walks = self._build_corpus(ins_name, suffix, len(nodes), walk_per_node, walk_length, weights)
            print('# of walks generated: {}'.format(len(walks)))
            print('training word2vec model ...')
            model = Word2Vec(walks, vector_size=dim, window=5, min_count=0, sg=1, hs=0, workers=2, epochs=5)
            if self.save:
                model.save(w2v_path)
        return self._gen_feature(model=model, n_pairs=len(model.wv))

    def _build_corpus(self, ins_name, suffix, n_nodes, walk_per_node, walk_length, weights):
        """
        build corpus based on the ancillary graph
        :param ins_name: instance name, str
        :param n_nodes: number of nodes in the ancillary graph
        :param walk_per_node: number of walks we want to generate for each node
        :param walk_length: maximum length for each walk
        :param weights: edge weights for the ancillary graph, n_nodes x n_nodes
        :return: a list of walks
        """
        corpus_path = './prob/{}/emb/{}/corpus_{}'.format(ins_name, self.variant_type, suffix)
        if os.path.exists(corpus_path):
            print('already generated, loading ...')
            with open(corpus_path, 'rb') as f:
                walks = pickle.load(f)
        else:
            nodes = np.arange(n_nodes)
            walks = []
            for _ in tqdm(range(walk_per_node)):
                np.random.shuffle(nodes)
                for n in nodes:
                    walk = self._random_walk(n, walk_length, weights)
                    walks.append(walk)
            if self.save:
                with open(corpus_path, 'wb') as f:
                    pickle.dump(walks, f)
        return walks

    @staticmethod
    def _random_walk(start, walk_length, weights):
        """
        generate a random walk from a stating node
        :param start: the starting node
        :param walk_length: the maximum length of the walk
        :param weights: the edge weights of the ancillary graph
        :return: a list of nodes in the generated walk
        """
        walk = [str(start)]
        curr = start
        while len(walk) - 1 < walk_length and len(weights[curr]) > 0:
            candidates = list(weights[curr].keys())
            w = np.array(list(weights[curr].values()))
            w /= w.sum()
            curr = np.random.choice(candidates, p=w)
            walk.append(str(curr))
        return walk

    @staticmethod
    def _gen_feature(model, n_pairs):
        return np.array([model.wv[str(i)] for i in range(n_pairs)])


class DeepWalk_utility:

    def __init__(self, random_seed=None, save=False):
        if random_seed:
            np.random.seed(random_seed)
        self.save = save

    def node2vec(self, ins_name, suffix, weights, walk_per_node=50, walk_length=10, dim=32):
        w2v_path = './prob/{}/emb/utility/emb_{}_dim{}'.format(ins_name, suffix, dim)
        if os.path.exists(w2v_path):
            print('already trained, loading w2v model ...')
            model = Word2Vec.load(w2v_path)
        else:
            nodes = range(len(weights))
            print('number of paths to sample: {}'.format(len(nodes) * walk_per_node))
            print('sampling paths ...')
            walks = self._build_corpus(ins_name, suffix, len(nodes), walk_per_node, walk_length, weights)
            print('# of walks generated: {}'.format(len(walks)))
            print('training word2vec model ...')
            model = Word2Vec(walks, vector_size=dim, window=5, min_count=0, sg=1, hs=0, workers=2, epochs=5)
            if self.save:
                model.save(w2v_path)
        return self._gen_feature(model=model, n_pairs=len(model.wv))

    def _build_corpus(self, ins_name, suffix, n_nodes, walk_per_node, walk_length, weights):
        """
        build corpus based on the ancillary graph
        :param ins_name: instance name, str
        :param n_nodes: number of nodes in the ancillary graph
        :param walk_per_node: number of walks we want to generate for each node
        :param walk_length: maximum length for each walk
        :param weights: edge weights for the ancillary graph, n_nodes x n_nodes
        :return: a list of walks
        """
        corpus_path = './prob/{}/emb/utility/corpus_{}'.format(ins_name, suffix)
        if os.path.exists(corpus_path):
            print('already generated, loading ...')
            with open(corpus_path, 'rb') as f:
                walks = pickle.load(f)
        else:
            nodes = np.arange(n_nodes)
            walks = []
            for _ in tqdm(range(walk_per_node)):
                np.random.shuffle(nodes)
                for n in nodes:
                    walk = self._random_walk(n, walk_length, weights)
                    walks.append(walk)
            if self.save:
                with open(corpus_path, 'wb') as f:
                    pickle.dump(walks, f)
        return walks

    @staticmethod
    def _random_walk(start, walk_length, weights):
        """
        generate a random walk from a stating node
        :param start: the starting node
        :param walk_length: the maximum length of the walk
        :param weights: the edge weights of the ancillary graph
        :return: a list of nodes in the generated walk
        """
        walk = [str(start)]
        curr = start
        while len(walk) - 1 < walk_length and len(weights[curr]) > 0:
            candidates = list(weights[curr].keys())
            w = np.array(list(weights[curr].values()))
            w /= w.sum()
            curr = np.random.choice(candidates, p=w)
            walk.append(str(curr))
        return walk

    @staticmethod
    def _gen_feature(model, n_pairs):
        return np.array([model.wv[str(i)] for i in range(n_pairs)])


class DeepWalk:

    def __init__(self, random_seed=None, save=False):
        if random_seed:
            np.random.seed(random_seed)
        self.save = save

    def node2vec(self, ins_name, suffix, weights, walk_per_node=50, walk_length=10, dim=32):
        w2v_path = './prob/{}/emb/emb_{}_dim{}'.format(ins_name, suffix, dim)
        if os.path.exists(w2v_path):
            print('already trained, loading w2v model ...')
            model = Word2Vec.load(w2v_path)
        else:
            nodes = range(len(weights))
            print('number of paths to sample: {}'.format(len(nodes) * walk_per_node))
            print('sampling paths ...')
            walks = self._build_corpus(ins_name, suffix, len(nodes), walk_per_node, walk_length, weights)
            print('# of walks generated: {}'.format(len(walks)))
            print('training word2vec model ...')
            model = Word2Vec(walks, vector_size=dim, window=5, min_count=0, sg=1, hs=0, workers=2, epochs=5)
            if self.save:
                model.save(w2v_path)
        return self._gen_feature(model=model, n_pairs=len(model.wv))

    def _build_corpus(self, ins_name, suffix, n_nodes, walk_per_node, walk_length, weights):
        """
        build corpus based on the ancillary graph
        :param ins_name: instance name, str
        :param n_nodes: number of nodes in the ancillary graph
        :param walk_per_node: number of walks we want to generate for each node
        :param walk_length: maximum length for each walk
        :param weights: edge weights for the ancillary graph, n_nodes x n_nodes
        :return: a list of walks
        """
        corpus_path = './prob/{}/emb/corpus_{}'.format(ins_name, suffix)
        if os.path.exists(corpus_path):
            print('already generated, loading ...')
            with open(corpus_path, 'rb') as f:
                walks = pickle.load(f)
        else:
            nodes = np.arange(n_nodes)
            walks = []
            for _ in tqdm(range(walk_per_node)):
                np.random.shuffle(nodes)
                for n in nodes:
                    walk = self._random_walk(n, walk_length, weights)
                    walks.append(walk)
            if self.save:
                with open(corpus_path, 'wb') as f:
                    pickle.dump(walks, f)
        return walks

    @staticmethod
    def _random_walk(start, walk_length, weights):
        """
        generate a random walk from a stating node
        :param start: the starting node
        :param walk_length: the maximum length of the walk
        :param weights: the edge weights of the ancillary graph
        :return: a list of nodes in the generated walk
        """
        walk = [str(start)]
        curr = start
        while len(walk) - 1 < walk_length and len(weights[curr]) > 0:
            candidates = list(weights[curr].keys())
            w = np.array(list(weights[curr].values()))
            w /= w.sum()
            curr = np.random.choice(candidates, p=w)
            walk.append(str(curr))
        return walk

    @staticmethod
    def _gen_feature(model, n_pairs):
        return np.array([model.wv[str(i)] for i in range(n_pairs)])


def build_ancillary_graph(ins_name, suffix, conn_matrix, threshold, time_matrix=[], save=False):
    weight_path = './prob/{}/emb/ancillary_graph_weights_{}.pkl'.format(ins_name, suffix)
    if os.path.exists(weight_path):
        print('loading ...')
        with open(weight_path, 'rb') as f:
            weights = pickle.load(f)
    else:
        # build ancillary graph from files
        print('building the relationship graph ...')
        if len(time_matrix) == 0:
            n_conn = conn_matrix.sum(axis=0)
            weights = [{} for _ in range(conn_matrix.shape[1])]
            count = 0
            # add a small number in case there exists a 0 entry
            pi_mat = np.dot(conn_matrix.T, conn_matrix)
            pi_mat = pi_mat / (np.tile(n_conn.reshape(-1, 1), (1, n_conn.shape[0])) +
                               np.tile(n_conn.reshape(1, -1), (n_conn.shape[0], 1)) -
                               pi_mat + 1e-2)
            for i in range(pi_mat.shape[0]):
                for j in range(i):
                    if pi_mat[i, j] >= threshold:
                        weights[i][j] = pi_mat[i, j]
                        weights[j][i] = pi_mat[i, j]
                        count += 1
        else:
            m, n = time_matrix.shape
            weights = np.zeros((n, n))
            beta = [1, 0.001, (1 - 0.06) / 10]
            # fill in negative numbers
            flag = time_matrix < 0
            time_matrix = time_matrix * (1 - flag) + time_matrix.max() * flag
            # calculate acc
            extra = time_matrix - 60
            flag = extra > 0
            acc = beta[0] - time_matrix * beta[1] - extra * flag * (beta[2] - beta[1])
            # construct the relationship graph
            for i in tqdm(range(n)):
                for j in range(i):
                    v1 = acc[:, i]
                    v2 = acc[:, j]
                    pi = np.minimum(v1 / (v2 + 0.001), v2 / (v1 + 0.001)).sum()
                    if pi > 0:
                        weights[i][j] = pi
                        weights[j][i] = pi
            weights = [{j: weights[i, j] for j in np.argsort(weights[i])[-50:] if weights[i, j]} for i in range(n)]
        # print('The ancillary graph consists of {} nodes and {} edges'.format(len(n_conn), count * 2))
        if save:
            with open(weight_path, 'wb') as f:
                pickle.dump(weights, f)
    return weights


def build_ancillary_graph_variant(ins_name, suffix, time_matrix, save=False, variant_type='exp'):
    betas = {'exp': [1, 0.75/20, 0.25/40], 'linear': [1, 1/60, 0], 'rec': [1, 0.001, 0.942/2]}
    thres = {'exp': 20, 'linear': 60, 'rec': 58}
    if variant_type not in betas:
        raise ValueError('the specified variant type {} not found'.format(variant_type))
    weight_path = './prob/{}/emb/{}/ancillary_graph_weights_{}.pkl'.format(ins_name, variant_type, suffix)
    if os.path.exists(weight_path):
        print('loading ...')
        with open(weight_path, 'rb') as f:
            weights = pickle.load(f)
    else:
        # build ancillary graph from files
        print('building the relationship graph ...')
        m, n = time_matrix.shape
        weights = np.zeros((n, n))
        # beta = [1, 0.001, (1 - 0.06) / 10]
        beta = betas[variant_type]
        # fill in negative numbers
        flag = time_matrix < 0
        time_matrix = time_matrix * (1 - flag) + time_matrix.max() * flag
        time_matrix = np.minimum(time_matrix, 60)
        # calculate acc
        extra = time_matrix - thres[variant_type]
        flag = extra > 0
        acc = beta[0] - time_matrix * beta[1] - extra * flag * (beta[2] - beta[1])
        # construct the relationship graph
        for i in tqdm(range(n)):
            for j in range(i):
                v1 = acc[:, i]
                v2 = acc[:, j]
                pi = np.minimum(v1 / (v2 + 0.001), v2 / (v1 + 0.001)).sum()
                if pi > 0:
                    weights[i][j] = pi
                    weights[j][i] = pi
        weights = [{j: weights[i, j] for j in np.argsort(weights[i])[-50:] if weights[i, j]} for i in range(n)]
        # print('The ancillary graph consists of {} nodes and {} edges'.format(len(n_conn), count * 2))
        if save:
            with open(weight_path, 'wb') as f:
                pickle.dump(weights, f)
    return weights


def build_ancillary_graph_utility(ins_name, suffix, utility_matrix, n_neighbor=10, save=False):
    weight_path = './prob/{}/emb/utility/ancillary_graph_weights_{}.pkl'.format(ins_name, suffix)
    if os.path.exists(weight_path):
        print('loading ...')
        with open(weight_path, 'rb') as f:
            weights = pickle.load(f)
    else:
        # build ancillary graph from files
        n = utility_matrix.shape[1]
        std_vec = utility_matrix.std(axis=0, keepdims=True)
        min_vec = utility_matrix.min(axis=0, keepdims=True)
        inc_matrix = (utility_matrix / min_vec - 1) / std_vec
        weights = np.zeros((n, n)) #  + np.eye(n) * 1e5
        for i in range(n):
            for j in range(i):
                v1 = inc_matrix[:, i]
                v2 = inc_matrix[:, j]
                num = ((v1 > 0) * (v2 > 0)).sum()
                pi = np.minimum(v1/(v2+0.001), v2/(v1+0.001)).sum() / ((v1 > 0).sum() + (v2 > 0).sum() - num + 0.001)
                weights[i, j] = pi
                weights[j, i] = pi
        weights = [{j: weights[i, j]
                    for j in np.argsort(weights[i])[-n_neighbor:]
                    if weights[i, j] > 0}
                   for i in range(n)]
        print('The ancillary graph consists of {} nodes and {} edges'.format(n, n * n_neighbor))
        if save:
            with open(weight_path, 'wb') as f:
                pickle.dump(weights, f)
    return weights


def build_ancillary_graph_route(ins_name, suffix, utility_matrix, n_neighbor=5, threshold=0.7, save=False):
    weight_path = './prob/{}/emb/ancillary_graph_weights_{}.pkl'.format(ins_name, suffix)
    if os.path.exists(weight_path):
        print('loading ...')
        with open(weight_path, 'rb') as f:
            weights = pickle.load(f)
    else:
        # build ancillary graph from files
        n = utility_matrix.shape[1]
        # weights = np.abs(np.corrcoef(utility_matrix, rowvar=False))
        # weights -= np.eye(n)
        # flag = (weights >= threshold).astype(int)
        # weights *= flag
        # inc_matrix = utility_matrix / utility_matrix.min(axis=0, keepdims=True) - 1
        std_vec = utility_matrix.std(axis=0, keepdims=True)
        min_vec = utility_matrix.min(axis=0, keepdims=True)
        inc_matrix = (utility_matrix / min_vec - 1) / std_vec
        weights = np.zeros((n, n)) #  + np.eye(n) * 1e5
        for i in range(n):
            for j in range(i):
                v1 = inc_matrix[:, i]
                v2 = inc_matrix[:, j]
                # pi = np.abs(v1 - v2).sum()
                # pi = (v1 * v2 * 100).sum() / (v1.sum() * v2.sum())
                num = ((v1 > 0) * (v2 > 0)).sum()
                pi = np.minimum(v1/(v2+0.001), v2/(v1+0.001)).sum() / ((v1 > 0).sum() + (v2 > 0).sum() - num + 0.001)
                weights[i, j] = pi
                weights[j, i] = pi
        # weights = [{j: 200/weights[i, j] for j in np.argsort(weights[i])[:n_neighbor] if weights[i, j]} for i in range(n)]
        weights = [{j: weights[i, j] for j in np.argsort(weights[i])[-n_neighbor:] if weights[i, j]} for i in range(n)]
        print('The ancillary graph consists of {} nodes and {} edges'.format(n, n * n_neighbor))
        if save:
            with open(weight_path, 'wb') as f:
                pickle.dump(weights, f)
    return weights


def add_position_feature(args, feature, shortest_path=True):
    destinations = args['destinations'] if 'sp_dist' not in args else args['sp_dist']
    od_pairs = des2od(destinations)
    coords = args['coordinates']
    center = np.array([coords[:, 0].max()//1, coords[:, 1].max()//1]) / 2
    loc_feature = []
    for orig, des in od_pairs:
        orig_lat, orig_lon = coords[orig]
        des_lat, des_lon = coords[des]
        c2o, c2d = one2all_dist(coords[[orig, des]], center)[0]
        od = one2all_dist(coords[[orig]], coords[des])[0][0]
        diff_lat, diff_lon = np.abs(orig_lat - des_lat), np.abs(orig_lon - des_lon)
        area = diff_lat * diff_lon
        if shortest_path:
            sp_dist = destinations[orig][des]
            loc_feature.append([orig_lat, orig_lon, des_lat, des_lon, c2o, c2d, od, area, sp_dist])
        else:
            loc_feature.append([orig_lat, orig_lon, des_lat, des_lon, c2o, c2d, od, area])
    loc_feature = np.array(loc_feature)
    loc_feature = (loc_feature - loc_feature.min(axis=0, keepdims=1)) / (loc_feature.max(axis=0, keepdims=1) - loc_feature.min(axis=0, keepdims=1))
    loc_feature -= 0.5
    loc_feature *= 4
    feature = np.concatenate([feature, loc_feature], axis=1)
    return feature


def add_distributional_feature(feature, utility_matrix):
    follower_min = utility_matrix.min(axis=0, keepdims=True)
    follower_std = utility_matrix.std(axis=0, keepdims=True)
    feature = np.concatenate([follower_min.T, follower_std.T, feature * follower_std.T], axis=1)
    return standardize_feature(feature=feature)


def standardize_feature(feature):
    scaler = StandardScaler()
    scaler.fit(feature)
    return scaler.transform(feature)
