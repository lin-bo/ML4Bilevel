import cupy as np
from tqdm import tqdm

from utils.functions import load_file, dump_file


def load_matrix(time_mat_name):
    time_matrix, _ = load_file(f'./prob/trt/emb/time_matrix/{time_mat_name}.pkl')
    return np.asarray(time_matrix, dtype=np.float32)


def cal_weights(time_mat_name, chunk_size):
    # process time matrix
    # time_matrix = cp.concatenate([load_matrix(sn) for j in range(sn, sn + 2)], axis=0)
    time_matrix = load_matrix(time_mat_name)
    n_sim, n_od = time_matrix.shape
    # convert to accessibility
    beta = np.asarray([1, 0.001, (1 - 0.001 * 28)/2], dtype=np.float32)
    time_matrix = np.minimum(time_matrix, 30)
    time_matrix = beta[0] - time_matrix * beta[1] - np.maximum(time_matrix - 28, 0) * (beta[2] - beta[1])
    time_matrix = time_matrix * (time_matrix <= 1)
    # initialize the weight dict
    weight = {}
    n_chunk = int(n_od / chunk_size) + 1
    chunks = [list(range(idx * chunk_size, np.min([(idx + 1) * chunk_size, n_od]))) for idx in range(n_chunk)]
    for idx in tqdm(range(len(chunks))):
        chunk = chunks[idx]
        cols = np.transpose(np.take(time_matrix.T, np.expand_dims(np.asarray(chunk), axis=1), axis=0), (0, 2, 1))
        ratio_matrix = np.minimum(cols/(time_matrix + 0.0001), time_matrix /(cols + 0.0001)).sum(axis=1)
        nn = np.argsort(ratio_matrix, axis=1)[:, -50:]
        vals = np.sort(ratio_matrix, axis=1)[:, -50:]
        for num, c in enumerate(chunk):
            weight[c] = {j: val for j, val in zip(nn[num], vals[num])}
        break
    dump_file(f'./prob/trt/emb/weight_ratio/weight-{time_mat_name}.pkl', weight)
    return weight


cal_weights(time_mat_name='time_matrix-p500-u100-n2', chunk_size=1000)
