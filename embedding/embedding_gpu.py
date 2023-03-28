import numpy as np
import torch
from gensim.models import Word2Vec
import multiprocessing
from utils.functions import des2od, one2all_dist, cal_od_utility, dump_file, load_file
from tqdm import tqdm
import os
import pickle


def merge_time_matrix(ins_name, sidx, eidx):
    time_matrix, _ = load_file('./prob/{}/emb/time_matrix/time_matrix-p500-u100-chunk{}.pkl'.format(ins_name, sidx))
    for idx in tqdm(range(sidx + 1, eidx)):
        mat, _ = load_file('./prob/{}/emb/time_matrix/time_matrix-p500-u100-chunk{}.pkl'.format(ins_name, idx))
        time_matrix += mat
    return time_matrix


def compute_weights(time_matrix, step_size):
    time_matrix = (time_matrix > 0).to(torch.float32)
    d, n_od = time_matrix.shape
    n_step = int(n_od/step_size) + int(n_od % step_size > 0)
    weights = {}
    for idx in tqdm(range(n_step)):
        min_idx, max_idx = idx * step_size, np.min([n_od, (idx + 1) * step_size])
        pi_mat = torch.matmul(time_matrix[:, min_idx: max_idx].T, time_matrix)
        weights.update(generate_weights(pi_mat, min_idx))
    return weights


def generate_weights(pi_mat, min_idx):
    nn_idx = torch.argsort(pi_mat, dim=1)[:, -50:]
    pi_mat = torch.gather(input=pi_mat, dim=1, index=nn_idx)
    size, n_od = pi_mat.shape
    weight = {min_idx + idx: {nn.item(): pi_mat[idx, j].item() for j, nn in enumerate(nn_idx[idx]) if pi_mat[idx, j].item() > 0}
              for idx in range(size)}
    return weight


def build_ancillary_graph_gpu(ins_name, suffix, sidx, eidx, step_size, device='cuda'):
    weight_path = './prob/{}/emb/weights/weights-{}_{}-{}.pkl'.format(ins_name, suffix, sidx, eidx)
    if os.path.exists(weight_path):
        print('loading ...')
        with open(weight_path, 'rb') as f:
            weights = pickle.load(f)
    else:
        # load time matrix
        print('loading time matrices ...')
        time_matrix = merge_time_matrix(ins_name, sidx, eidx)
        time_matrix = torch.tensor(time_matrix, device=torch.device(device), dtype=torch.float32).detach()
        # compute weights
        print('computing weights')
        weights = compute_weights(time_matrix, step_size)
    dump_file(weight_path, weights)
    return weights


def update_weights(weights, w):
    for key, nns in tqdm(list(w.items())):
        if key in weights:
            for nn, val in nns.items():
                if nn in weights[key]:
                    weights[key][nn] += val
                else:
                    weights[key][nn] = val
        else:
            weights[key] = nns.copy()
    return weights


def merge_weights(sidx, eidx, step_size):
    weights = {}
    n_step = int((eidx - sidx) / step_size)
    for idx in range(n_step):
        print(idx)
        name = 'weights-p500-u100_{}-{}'.format(idx * step_size, (idx + 1) * step_size)
        w, _ = load_file('./prob/trt/emb/weights/{}.pkl'.format(name))
        weights = update_weights(weights, w)
    dump_file('./prob/trt/emb/weights/weights.pkl', weights)


class DeepWalk:

    def __init__(self, random_seed=None, save=False):
        if random_seed:
            np.random.seed(random_seed)
        self.save = save

    def node2vec(self, ins_name, suffix, weights, walk_per_node=50, walk_length=10, dim=32):
        w2v_path = './prob/{}/emb/emb_{}'.format(ins_name, suffix)
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
            model = Word2Vec(walks, vector_size=dim, window=5, min_count=0, sg=1, hs=0, workers=1)
            if self.save:
                model.save(w2v_path)
        return self._gen_feature(model=model, n_pairs=len(model.wv))

    def _build_corpus(self, ins_name, suffix, n_nodes, walk_per_node, walk_length, weights):
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


def random_walk(start, walk_length, weights):
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
        if w.sum() == 0:
            break
        w /= w.sum()
        curr = np.random.choice(candidates, p=w)
        walk.append(str(curr))
    return walk


def build_corpus_one_round(ins_name, walk_length, round_idx, weight_file_name):
    corpus_path = './prob/{}/emb/corpus/corpus_{}'.format(ins_name, round_idx)
    if os.path.exists(corpus_path):
        print('already generated, loading')
    else:
        print('round {}'.format(round_idx), corpus_path)
        weights, _ = load_file(f'./prob/trt/emb/weight_ratio/{weight_file_name}.pkl')
        nodes = list(weights.keys())
        walks = []
        for n in tqdm(nodes):
            walk = random_walk(n, walk_length, weights)
            if len(walk) <= 1:
                continue
            else:
                walks.append(walk)
        with open(corpus_path, 'wb') as f:
            pickle.dump(walks, f)


def build_corpus(ins_name, walk_length, walk_per_node, n_workers, weight_file_name):
    print('start building corpus, number of workers: {}'.format(n_workers))
    # generate parameters for multi-processing
    params = []
    for round_idx in list(range(walk_per_node))[::-1]:
        params.append((ins_name, walk_length, round_idx, weight_file_name))
    pool = multiprocessing.Pool(n_workers)
    _ = pool.starmap(build_corpus_one_round, params)
    pool.close()


class SentenceIterator:

    def __init__(self, suffix=''):
        self.suffix = suffix

    def __iter__(self):
        """
        Generator: iterate over all relevant documents, yielding one
        document (=list of utf8 tokens) at a time.
        """
        for idx in range(50):
            # read each document as one big string
            corpus, _ = load_file('./prob/trt/emb/corpus{}/corpus_{}'.format(self.suffix, idx))
            for sentence in corpus:
                yield sentence


def deep_walk_training(dim=32, n_workers=8, suffix=''):
    w2v_path = './prob/trt/emb/emb{}'.format(suffix)
    if os.path.exists(w2v_path):
        print('already trained, loading w2v model ...')
        model = Word2Vec.load(w2v_path)
    else:
        print('start training')
        iter_sentence = SentenceIterator(suffix=suffix)
        model = Word2Vec(iter_sentence,
                         vector_size=dim, window=5, min_count=0,
                         workers=n_workers, epochs=10)
        model.save(w2v_path)
    # fea = np.array([model.wv[str(idx)] for idx in range(1327849) if str(idx) in model.wv])
    # pairs = [idx for idx in range(1327849) if str(idx) in model.wv]
    fea = np.array([model.wv[str(idx)] for idx in range(1360450) if str(idx) in model.wv])
    pairs = [idx for idx in range(1360450) if str(idx) in model.wv]
    dump_file('./prob/trt/emb/emb{}.pkl'.format(suffix), fea)
    dump_file('./prob/trt/emb/emb_pairs{}.pkl'.format(suffix), pairs)
    return fea, pairs
