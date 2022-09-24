#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

import functools
import operator
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path_length
import numpy as np
import pandas as pd
import pickle
import os
from tqdm import tqdm
from scipy.spatial import distance_matrix


def flatten(lists):
    """
    flatten the given list
    :param lists: a list of lists
    :return: a list of all the elements in these lists
    """
    return functools.reduce(operator.iconcat, lists, [])


def des2od(destinations):
    """
    convert dict of destinations to a list of od pairs
    :param destinations: {orig: [destinations]}
    :return: [od pairs]
    """
    return [(orig, des) for orig, destination in destinations.items() for des in destination]


def cal_acc(args, new_projects, new_signals, impedence='travel_time', potential='populations', by_orig=False):
    """
    calculate the accessibility of the instance given the selected projects and signals
    :param args:
    :param new_projects:
    :param new_signals:
    :return: accessibility
    """
    # retrieve information
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
    edges_w_attr = [(i, j, {impedence: travel_time[i, j]}) for (i, j) in new_edges]
    # add new edges
    G_curr.add_edges_from(edges_w_attr)
    acc = 0
    acc_by_orig = {}
    for orig in tqdm(destinations):
        orig_acc = 0
        lengths = single_source_dijkstra_path_length(G=G_curr, source=orig, cutoff=T, weight=impedence)
        reachable_des = [des for des in lengths if des in destinations[orig]]
        for des in reachable_des:
            acc += pop[des]
            orig_acc += pop[des]
        acc_by_orig[orig] = orig_acc
    if by_orig:
        return acc, acc_by_orig
    return acc


def gen_betas(beta_1, T, M):
    beta_0 = 1
    beta_2 = (beta_0 - beta_1 * T) / (M - T)
    return [beta_0, beta_1, beta_2]


def penalty(t, beta, T, M):
    if t <= T:
        p = t * beta[1]
    elif t <= M:
        p = beta[1] * T + beta[2] * (t - T)
    else:
        p = beta[0]
    return p


def find_remaining_pairs(args, new_projects, new_signals, beta, impedence='travel_time'):
    """
    calculate the accessibility of the instance given the selected projects and signals
    :param args:
    :param new_projects:
    :param new_signals:
    :return: accessibility
    """
    # retrieve information
    G_curr = args['G_curr'].copy()
    G = args['G'].copy()
    T = args['travel_time_limit']
    M = args['travel_time_max']
    pop = args['populations']
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
    edges_w_attr = [(i, j, {impedence: travel_time[i, j]}) for (i, j) in new_edges]
    # add new edges
    G_curr.add_edges_from(edges_w_attr)
    cnt = -1
    remain_pairs = []
    new_destinations = {}
    for orig in tqdm(destinations):
        lengths = single_source_dijkstra_path_length(G=G_curr, source=orig, cutoff=M, weight=impedence)
        reachable_des = {des: lengths[des] for des in lengths if des in destinations[orig]}
        tmp = {}
        for des in destinations[orig]:
            cnt += 1
            if des not in reachable_des:
                remain_pairs.append(cnt)
                tmp[des] = destinations[orig][des]
        new_destinations[orig] = tmp.copy()
    return remain_pairs, new_destinations


def cal_con_obj(args, new_projects, new_signals, beta, impedence='travel_time', manual_params={}):
    """
    calculate the accessibility of the instance given the selected projects and signals
    :param args:
    :param new_projects:
    :param new_signals:
    :return: accessibility
    """
    # retrieve information
    G_curr = args['G_curr'].copy()
    G = args['G'].copy()
    T = args['travel_time_limit']
    M = args['travel_time_max']
    pop = args['populations']
    destinations = args['destinations']
    projs = args['projects']
    travel_time = args['travel_time']
    unsig_inters = args['signal_costs']
    # set manual params
    if len(manual_params) > 0:
        T, M, beta = manual_params['T'], manual_params['M'], manual_params['beta']
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
    edges_w_attr = [(i, j, {impedence: travel_time[i, j]}) for (i, j) in new_edges]
    # add new edges
    G_curr.add_edges_from(edges_w_attr)
    obj = 0
    acc = 0
    penalties = {}
    max_obj = 0
    for orig in destinations:
        lengths = single_source_dijkstra_path_length(G=G_curr, source=orig, cutoff=M, weight=impedence)
        reachable_des = {des: lengths[des] for des in lengths if des in destinations[orig]}
        for des in destinations[orig]:
            penalties[(orig, des)] = penalty(reachable_des[des], beta, T, M) if des in reachable_des else 0
            if des in reachable_des:
                obj += penalty(reachable_des[des], beta, T, M) * pop[des]
                acc += pop[des]
            else:
                obj += pop[des]
            max_obj += pop[des]
    obj = max_obj - obj
    return obj, acc, penalties


def cal_utility_obj(args, new_projects, alpha=1.02):
    # extract parameters
    od_routes = args['od_routes']
    v_bar = args['v_bar']
    pop = args['populations']
    n_orig = len(od_routes)
    # get edges along new project
    new_proj_edges = []
    for idx in new_projects:
        new_proj_edges += args['projects'][idx]
    new_proj_edges = set(new_proj_edges)
    # calculate the objective
    obj = 0
    for orig, destinations in od_routes.items():
        for des, routes in destinations.items():
            u = cal_od_utility(routes, v_bar[orig][des], new_proj_edges, n_orig, alpha)
            obj += pop[des] * u
    return obj


def cal_od_utilies(args, new_projects, alpha=1.02):
    # extract parameters
    od_routes = args['od_routes']
    v_bar = args['v_bar']
    n_orig = len(od_routes)
    # get edges along new project
    new_proj_edges = []
    for idx in new_projects:
        new_proj_edges += args['projects'][idx]
    new_proj_edges = set(new_proj_edges)
    # calculate the objective
    utilities = {}
    for orig, destinations in od_routes.items():
        for des, routes in destinations.items():
            utilities[orig, des] = cal_od_utility(routes, v_bar[orig][des], new_proj_edges, n_orig, alpha)
    return utilities


def cal_od_utility(routes, v_bars, proj_edges, n_orig, alpha):
    v = [cal_route_utility(r, proj_edges, n_orig, alpha) for r in routes]
    v = np.array(v) - np.array(v_bars)
    v_exp = np.exp(v)
    prob = v_exp / v_exp.sum()
    return (prob * v).sum()


def cal_route_utility(route, proj_edges, n_orig, alpha):
    u = 0
    cont_len = 0
    for i in range(len(route) - 1):
        if ((route[i], route[i+1]) in proj_edges) or (route[i] < n_orig) or (route[i+1] < n_orig):
            cont_len += 1
        else:
            u += cont_len * (alpha ** cont_len)
            cont_len = 0
    u += cont_len * (alpha ** cont_len)
    return u


def one2all_dist(vecs, vec):
    """
    calculate the distance from a node to each node in a given set
    :param vecs: n x d array, each row is a node
    :param vec: d, array, representing the node of interests
    :return: 1, n array, representing the distances
    """
    return np.sqrt(((vecs - vec) ** 2).sum(axis=1)).reshape((1, -1))


def add_intercept_col(feature):
    n, _ = feature.shape
    return np.concatenate([np.ones((n, 1)), feature], axis=1)


def cal_od_stds(utility_matrix, od_pairs):
    std_vec = utility_matrix.std(axis=0)
    std_dict = {}
    for idx in range(len(od_pairs)):
        std_dict[od_pairs[idx][0], od_pairs[idx][1]] = std_vec[idx]
    return std_dict


def load_file(path):
    if os.path.exists(path):
        if path[-4:] == '.csv':
            file = pd.read_csv(path)
        else:
            with open(path, 'rb') as f:
                file = pickle.load(f)
                f.close()
        return file, True
    else:
        return None, False


def dump_file(path, file):
    if path[-4:] == '.csv':
        file.to_csv(path, index=False)
    else:
        with open(path, 'wb') as f:
            pickle.dump(file, f)
            f.close()


def proj2nodes(project):
    nodes = set([])
    for (fnode, tnode) in project:
        nodes.add(fnode)
        nodes.add(tnode)
    return nodes


def str2list(s):
    l = s[1:-1].strip().split(',')
    return [int(i) for i in l]


def pairwise_distance(points_1, points_2, metric):
    if metric == 'euc':
        return distance_matrix(points_1, points_2)
    elif metric == 'cosine':
        dist_mat = np.dot(points_1, points_2.T) \
                   / np.sqrt((points_1 ** 2).sum(axis=1)).reshape((-1, 1)) \
                   / np.sqrt((points_2 ** 2).sum(axis=1)).reshape((1, -1))
        dist_mat = 1 - dist_mat
        return dist_mat
    else:
        raise ValueError('metric {} is not defined'.format(metric))


def time2acc_synthetic(time_matrix, variant_type):
    betas = {'exp': [1, 0.75 / 20, 0.25 / 40], 'linear': [1, 1 / 60, 0], 'rec': [1, 0.001, 0.942 / 2]}
    thres = {'exp': 20, 'linear': 60, 'rec': 58}
    beta = betas[variant_type]
    # fill in negative numbers
    flag = time_matrix < 0
    time_matrix = time_matrix * (1 - flag) + time_matrix.max() * flag
    time_matrix = np.minimum(time_matrix, 60)
    # calculate acc
    extra = time_matrix - thres[variant_type]
    flag = extra > 0
    acc = beta[0] - time_matrix * beta[1] - extra * flag * (beta[2] - beta[1])
    return acc
