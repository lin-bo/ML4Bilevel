#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path_length, single_source_dijkstra_path
import pickle
from utils.check import file_existence
from scipy.spatial import distance_matrix
from utils.functions import load_file, dump_file, des2od
from tqdm import tqdm
import pandas as pd
# import geopandas as gpd
# from shapely.geometry import LineString


class ClusterGridGenerator:

    def __init__(self, width, n_orig, discrete=True, time_limit=30, time_max=40, p_sig=0.3, p_orig_inter=0.7, n_inter=3, random_seed=None):
        '''
        A class that generates clustered grid instances
        where origin and destinations are clustered in each grid cell
        :param width: grid width
        :param n_orig: the number origins (each origin is also a destination)
        :param T: travel time limit
        :param p_signal: the % of intersections that have a traffic signal
        :param p_orig_inter: the % of connections established for each origin with its surrounding intersections
        :param n_inter: number of intersections along each grid edge
        :param random_seed: random seed for instance generation
        '''

        self.width = width
        self.n_orig = n_orig
        self.time_limit = time_limit
        self.time_max = time_max
        self.p_sig = p_sig
        self.p_orig_inter = p_orig_inter
        self.n_inter = n_inter
        self.discrete = discrete
        if random_seed:
            np.random.seed(random_seed)

    def generate(self, save=False):
        # check if the instance has been generated or not
        if self.discrete:
            file_name = './prob/{}x{}-{}/args.pkl'.format(self.width, self.width, self.n_orig)
        else:
            file_name = './prob/{}x{}-{}/args_c.pkl'.format(self.width, self.width, self.n_orig)
        if file_existence(file_name):
            print('instance {} has been already been generated, loading from local drive ...'.format(file_name))
            with open(file_name, 'rb') as f:
                args = pickle.load(f)
        else:
            nodes, coords, sigs, cell2orig = self._gen_nodes()
            edges, projs = self._gen_edges(cell2orig=cell2orig)
            pop, travel_time, proj_cost, sig_cost = self._gen_features(edges=edges, projs=projs, sigs=sigs)
            edge2proj = self._gen_mapping(projs)
            od_connected, G_curr = self._get_connected_shortest_length(edges=edges, travel_time=travel_time, sigs=sigs)
            od_overall, G = self._get_overall_reachable_shortest_length(edges=edges, travel_time=travel_time)
            destinations = self._extract_od_pairs(od_overall=od_overall, od_connected=od_connected)
            args = {'destinations': destinations,
                    'populations': pop,
                    'G': G,
                    'G_curr': G_curr,
                    'n_nodes': len(nodes),
                    'projects': projs,
                    'project_costs': proj_cost,
                    'signal_costs': sig_cost,
                    'coordinates': coords,
                    'travel_time': travel_time,
                    'edge2proj': edge2proj,
                    'travel_time_limit': self.time_limit,
                    'travel_time_max': self.time_max}
            if save:
                with open(file_name, 'wb') as f:
                    pickle.dump(args, f)
        return args

    def generate_wroutes(self, save=False):
        # ser time limit and time max, they should be large because we want to ignore them
        # self.time_limit = 1e5
        # self.time_max = 1e5
        # check if the instance has been generated or not
        file_name = './prob/{}x{}-{}/args_r.pkl'.format(self.width, self.width, self.n_orig)
        if file_existence(file_name):
            print('instance {} has been already been generated, loading from local drive ...'.format(file_name))
            with open(file_name, 'rb') as f:
                args = pickle.load(f)
        else:
            nodes, coords, sigs, cell2orig = self._gen_nodes()
            edges, projs = self._gen_edges(cell2orig=cell2orig)
            pop, travel_time, proj_cost, sig_cost = self._gen_features(edges=edges, projs=projs, sigs=sigs)
            edge2proj = self._gen_mapping(projs)
            od_overall, G = self._get_overall_reachable_shortest_length(edges=edges, travel_time=travel_time, path=True)
            od_connected, _ = self._get_connected_shortest_length(edges=edges, travel_time=travel_time, sigs=sigs)
            od_remain = self._extract_od_pairs(od_overall=od_overall, od_connected=od_connected)
            routes = self._gen_candidate_routes(od_remain, coords, edges)
            destinations, seg2idx, segs = self._gen_continuous_segments(routes)
            segidx2proj = self._gen_segidx2proj(seg2idx=seg2idx, edge2proj=edge2proj)
            v_bar = self._cal_v_bar(routes)
            beta = self._cal_beta(alpha=1.02)
            args = {'destinations': destinations,
                    'od_routes': routes,
                    'seg2idx': seg2idx,
                    'segidx2proj': segidx2proj,
                    'segments': segs,
                    'v_bar': v_bar,
                    'beta': beta,
                    'populations': pop,
                    'G': G,
                    'n_nodes': len(nodes),
                    'projects': projs,
                    'project_costs': proj_cost,
                    'signal_costs': sig_cost,
                    'coordinates': coords,
                    'travel_time': travel_time,
                    'edge2proj': edge2proj,
                    'travel_time_limit': self.time_limit,
                    'travel_time_max': self.time_max,
                    'sp_dist': self._gen_sp_dist(G, destinations)}
            if save:
                with open(file_name, 'wb') as f:
                    pickle.dump(args, f)
        return args

    def visualize(self, args):

        coords, projs, edges, sigs = self._extract_args_for_visual(args)

        n_art_inter = (self.width + 1) ** 2 * 6
        origins = list(range(self.n_orig))
        inters = list(range(self.n_orig, len(coords)))
        sig_inter = [idx for idx in inters if idx not in sigs]
        unsig_inter = [idx for idx in inters if idx in sigs]

        fig, ax = plt.subplots(figsize=(10, 10))
        x, y = coords[:, 0], coords[:, 1]

        # plot arterial
        labeled = False
        for p in projs:
            ends = [p[0][0], p[-1][0]]
            if labeled:
                ax.plot(x[ends], y[ends], color='silver', linewidth=3)
            else:
                ax.plot(x[ends], y[ends], color='silver', linewidth=3, label='Arterial Road (high-stress)')
                labeled = True

        # plot connections between da and intersection
        labeled = False
        for edge in edges:
            if edge[0] in origins or edge[1] in origins:
                if labeled:
                    ax.plot(x[edge], y[edge], color='dimgrey')
                else:
                    ax.plot(x[edge], y[edge], color='dimgrey', label='Local Roads (low-stress)')
                    labeled = True

        # plot nodes in different colors
        ax.scatter(x[unsig_inter], y[unsig_inter], color='darkgray', zorder=10, label='Unsignalized Intersection')
        ax.scatter(x[sig_inter], y[sig_inter], color='black', zorder=10, label='Signalized Intersection')
        ax.scatter(x[origins], y[origins], marker='*', color='black', s=100, zorder=10, label='Population Centroid')

        # plot node names
        # for i in range(len(x)):
        #     ax.text(x[i] + 0.03, y[i] + 0.03, '{}'.format(i))

        ax.set_xticks([])
        ax.set_yticks([])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_visible(False)

        lines_labels = [ax.get_legend_handles_labels()]
        lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
        lgd = fig.legend(lines, labels, loc='center right', ncol=1, bbox_to_anchor=(1.25, 0.5), prop={'size': 15})

        plt.savefig('/Users/bolin/Desktop/figure.png', dpi=300, bbox_inches='tight')
        plt.show()

    @staticmethod
    def _extract_args_for_visual(args):
        coords = args['coordinates']
        projs = args['projects']
        sigs = args['signal_costs']
        edges = args['G'].edges
        return coords, projs, np.array(edges), sigs

    def _gen_nodes(self):
        """
        generate nodes in the network
        :return: a list of nodes, an array of coordinates (n_nodes x 2), an array of binary indicator, cell2orig ...
        """
        # generate origins
        # assign origins into cells,
        # cells numbered left -> right, top -> bottom, starting from zero
        sep = np.sort(np.append(np.random.choice(a=np.arange(1, self.n_orig),
                                                 size=self.width ** 2 - 1,
                                                 replace=False),
                                [0, self.n_orig]))
        origins = list(range(self.n_orig))
        cell2orig = [origins[sep[idx]: sep[idx + 1]] for idx in range(len(sep) - 1)]
        # generate coordinates
        coords = []
        for cell, origs in enumerate(cell2orig):
            coords += self._gen_orig_coord(cell, len(cell2orig[cell]))
        # generate arterial intersections (we assume they are all signalized)
        # Indices start from n_origins, left -> right, top -> bottom
        n_art_inter = (self.width + 1) ** 2
        for idx in range(n_art_inter):
            coords += [[idx % (self.width + 1), self.width - idx // (self.width + 1)]]
        # generate arterial & local intersections (we assume they may not be signalized),
        # they are ordered, horizontal - > vertical, left -> right, top -> bottom
        # intersections on horizontal edges
        for idx in range(self.width * (self.width + 1)):
            left_x = idx % self.width
            left_y = self.width - idx // self.width
            step = 1 / (self.n_inter + 1)
            coords += [[left_x + step * i, left_y] for i in range(1, self.n_inter + 1)]
        # intersections on vertical edges
        for idx in range(self.width * (self.width + 1)):
            low_x = idx % (self.width + 1)
            low_y = self.width - 1 - idx // (self.width + 1)
            step = 1 / (self.n_inter + 1)
            coords += [[low_x, low_y + step * i] for i in range(1, self.n_inter + 1)]
        # generate some useful lists
        coords = np.array(coords)
        nodes = np.arange(coords.shape[0])
        # generate signalized intersections and un-signalized intersections
        art_local_inter = np.arange(self.width * (self.width + 1) * self.n_inter * 2) + self.n_orig + n_art_inter
        n_unsig_inter = int(len(art_local_inter) * (1 - self.p_sig))
        unsig_inter = np.random.choice(art_local_inter, n_unsig_inter, replace=False)
        sigs = np.ones(self.n_orig + n_art_inter + len(art_local_inter))
        for i in unsig_inter:
            sigs[i] = 0
        return nodes, coords, sigs, cell2orig

    def _gen_orig_coord(self, cell, n_orig):
        '''
        generate the coordinates for origins in the given cell
        :param cell: the cell index
        :param n_orig: the number of origins in this cell
        :return: coordinates of the generated origins, array (n_orig, 2)
        '''
        # (0, 0) at the lower left corner
        coords = np.random.uniform(low=0.05, high=0.95, size=(n_orig, 2))

        # position the lower left corner at the right place
        x_shift = cell % self.width
        y_shift = self.width - 1 - cell // self.width

        coords += np.array([[x_shift, y_shift]])

        return coords.tolist()

    def _gen_edges(self, cell2orig):
        """
        generate the edges and the projects in the network
        :param cell2orig: list of origins in each grid cell
        :return: edges, and the edges in each project
        """
        # initialize record lists
        edges, projs = [], []
        cell_boundary = [[] for _ in range(self.width ** 2)]
        # generate horizontal edges between intersections
        for idx in range(self.width * (self.width + 1)):
            node_seq = self._horizontal_node_sequence(idx=idx)
            new_edges = self._edges_on_art(node_seq)
            edges += new_edges
            projs.append(new_edges)
            cells = self._horizontal_related_cells(idx=idx)
            for c in cells:
                cell_boundary[c] += node_seq[1: -1]
        # generate vertical edges between intersections
        for idx in range(self.width * (self.width + 1)):
            node_seq = self._vertical_node_sequence(idx=idx)
            new_edges = self._edges_on_art(node_seq)
            edges += new_edges
            projs.append(new_edges)
            cells = self._vertical_related_cells(idx=idx)
            for c in cells:
                cell_boundary[c] += node_seq[1: -1]
        # generate edges between DAs and intersections
        for cell in range(self.width ** 2):
            edge_inters = cell_boundary[cell]
            n_connect = int(len(edge_inters) * self.p_orig_inter)
            for orig in cell2orig[cell]:
                inters = np.random.choice(a=edge_inters, size=n_connect, replace=False)
                for inter in inters:
                    new_edges = [[orig, inter], [inter, orig]]
                    edges += new_edges
        return edges, projs

    def _horizontal_node_sequence(self, idx):
        """
        map the given index to the nodes along the corresponding segment
        :param idx: int, ordered from left to right, top to bottom
        :return: a list of nodes, ordered from left to right
        """
        n_art_inter = (self.width + 1) ** 2
        left_x = idx % self.width
        left_y = self.width - idx // self.width
        left_art_inter = (self.width - left_y) * (self.width + 1) + left_x + self.n_orig
        left_art_local_inter = idx * self.n_inter + n_art_inter + self.n_orig
        seq = [left_art_local_inter + i for i in range(self.n_inter)]
        node_seq = [left_art_inter] + seq + [left_art_inter + 1]
        return node_seq

    def _horizontal_related_cells(self, idx):
        """
        get the calls whose boundary contain this horizontal segment
        :param idx: the index of the horizontal segment
        :return: list of cell indices
        """
        cells = []
        n_cell = self.width ** 2
        if idx < n_cell:
            cells.append(idx)
        if idx >= self.width:
            cells.append(idx - self.width)
        return cells

    def _vertical_node_sequence(self, idx):
        """
        map the given index to the nodes along the corresponding segment
        :param idx: int, ordered from left to right, top to bottom
        :return: a list of nodes, ordered from low to high
        """
        n_art_inter = (self.width + 1) ** 2
        low_x = idx % (self.width + 1)
        low_y = self.width - 1 - idx // (self.width + 1)
        low_art_inter = (self.width + 1) * (self.width - low_y) + low_x + self.n_orig
        low_art_local_inter = idx * self.n_inter + n_art_inter + self.n_orig + \
                              self.width * (self.width + 1) * self.n_inter
        seq = [low_art_local_inter + i for i in range(self.n_inter)]
        node_seq = [low_art_inter] + seq + [low_art_inter - self.width - 1]
        return node_seq

    def _vertical_related_cells(self, idx):
        """
        get the calls whose boundary contain this vertical segment
        :param idx: the index of the vertical segment
        :return: list of cell indices
        """
        cells = []
        low_x = idx % (self.width + 1)
        low_y = self.width - 1 - idx // (self.width + 1)
        if low_x < self.width:
            cells.append(low_x + (self.width - 1 - low_y) * self.width)
        if low_x >= 1:
            cells.append(low_x + (self.width - 1 - low_y) * self.width - 1)
        return cells

    @staticmethod
    def _edges_on_art(node_seq):
        """
        add edges along each arterial segment
        :param node_seq: a (ordered) sequence of node along an arterial segment
        :return: a list of edges, each edge is represented by its two ends
        """
        new_edges = []
        for i in range(len(node_seq) - 1):
            new_edges.append((node_seq[i], node_seq[i + 1]))
            new_edges.append((node_seq[i + 1], node_seq[i]))
        return new_edges

    def _gen_features(self, edges, projs, sigs):
        """
        generate features
        :param n_edges: the number of edges
        :param projs: list of projects and its related edges
        :param sigs: binary list indicating if each node has traffic signals or not
        :return:
        """
        pop = {n: np.random.randint(low=1, high=10) for n in range(self.n_orig)}
        travel_time = {(i, j): np.random.randint(low=3, high=10) for (i, j) in edges}
        proj_cost = {idx: np.sum([travel_time[e] for e in proj]) for idx, proj in enumerate(projs)}
        sig_cost = {idx: np.random.randint(low=5, high=8) for idx, val in enumerate(sigs) if val == 0}
        return pop, travel_time, proj_cost, sig_cost

    @staticmethod
    def _cal_beta(alpha=1.02):
        beta = [0, alpha]
        f = [0, alpha]
        for i in range(2, 100):
            f_val = i * (alpha ** i)
            f.append(f_val)
            beta.append(f[-1] - 2 * f[-2] + f[-3])
        return beta

    @staticmethod
    def _gen_mapping(projs):
        """
        generate mappings
        :param projs:
        :return:
        """
        edge2proj = {(i, j): idx for idx, edges in enumerate(projs) for (i, j) in edges}
        return edge2proj

    def _gen_candidate_routes(self, od_remain, coords, edges):
        destinations = {orig: {des: [od_remain[orig][des]]
                               for des in od_remain[orig]}
                        for orig in od_remain}
        # generate paths based on euclidean cost
        euc_cost = distance_matrix(coords, coords)
        G_euc = self._construct_network(edges=edges, attrs={'travel_time': [euc_cost[i, j] for i, j in edges]})
        euc_paths = self._get_reachable_shortest_length(G=G_euc, cutoff=1e5, path=True)
        destinations = self._merge_path_dict(destinations, euc_paths)
        # generate paths favoring arterial roads
        art_costs = []
        for i, j in edges:
            c = euc_cost[i, j] if (i < self.n_orig) or (j < self.n_orig) else euc_cost[i, j] / 100
            art_costs.append(c)
        G_art = self._construct_network(edges=edges, attrs={'travel_time': art_costs})
        art_paths = self._get_reachable_shortest_length(G=G_art, cutoff=1e5, path=True)
        destinations = self._merge_path_dict(destinations, art_paths)
        return destinations

    def _gen_continuous_segments(self, routes):
        """
        :param routes: dict, {orig:{des: [[r1], [r2], [r3]], ...}, ...}
        :return:
        """
        # parameters
        cnt = -1
        seg2idx = {}
        segs = []
        destinations = {}
        # exam each route generated
        print('generating continuous segments along routes')
        for orig, dests in routes.items():
            orig_dests = {}
            for des, rs in dests.items():
                route_segments = []
                for i in range(3):
                    segments = self._gen_segments(rs[i])
                    seg2idx, idx_enroute, segs, cnt = self._add_segments(segments, seg2idx, segs, cnt)
                    route_segments.append(idx_enroute)
                orig_dests[des] = route_segments
            destinations[orig] = orig_dests
        print('{} continuous segments generated'.format(cnt))
        return destinations, seg2idx, segs

    def _gen_segments(self, route):
        """
        given a route (a list of nodes), generate the continuous segments along the route
        :param route: list of nodes
        :return: list of segments (list of nodes)
        """
        n = len(route)
        segments = []
        for i in range(2, n + 1):
            for j in range(n - i + 1):
                segments.append(route[j: j+i])
        return segments

    def _add_segments(self, segments, seg2idx, segs, cnt):
        idx_enroute = []
        for s in segments:
            key = tuple(s)
            if key not in seg2idx:
                cnt += 1
                seg2idx[key] = cnt
                segs.append(key)
                idx_enroute.append(cnt)
            else:
                idx_enroute.append(seg2idx[key])
        return seg2idx, idx_enroute, segs, cnt

    def _gen_segidx2proj(self, seg2idx, edge2proj):
        segidx2proj = {}
        for seg, idx in seg2idx.items():
            projs = set([])
            for j in range(len(seg) - 1):
                if (seg[j], seg[j+1]) in edge2proj:
                    projs.add(edge2proj[seg[j], seg[j+1]])
            segidx2proj[idx] = list(projs)
        return segidx2proj

    def _cal_v_bar(self, routes):
        v_bar = {}
        for orig, dests in routes.items():
            v_bar_orig = {}
            for des, rs in dests.items():
                v_bar_orig_des = []
                enroute_edges = [self._enroute_node2edge(r) for r in rs]
                for i in range(3):
                    ps = self.v_bar(enroute_edges[i], [enroute_edges[j] for j in range(3) if j != i])
                    v_bar_orig_des.append(np.log(ps))
                v_bar_orig[des] = v_bar_orig_des
            v_bar[orig] = v_bar_orig
        return v_bar

    @staticmethod
    def _enroute_node2edge(nodes):
        n = len(nodes)
        edges = set([])
        for i in range(n - 1):
            edges.add((nodes[i], nodes[i+1]))
        return edges

    @staticmethod
    def v_bar(target, others):
        val = 0
        frac = 1 / len(target)
        for edge in target:
            val += frac * 1 / (1 + np.sum([edge in o for o in others]))
        return val

    def _get_overall_reachable_shortest_length(self, edges, travel_time, path=False):
        """
        get the reachable destinations within the given time limit for each origin (on the overall network)
        :param edges: a list of edges (n_edges x 2)
        :param time: a list of travel time on each edge
        :return: a list of destinations for each origin
        """
        G = self._construct_network(edges=edges, attrs={'travel_time': [travel_time[i, j] for i, j in edges]})
        cutoff = self.time_limit if self.discrete else self.time_max
        return self._get_reachable_shortest_length(G=G, cutoff=cutoff, path=path), G

    def _get_connected_shortest_length(self, edges, travel_time, sigs):
        # get traversable origin-inter edges
        n_inter_inter_edges = self.width * (self.width + 1) * (self.n_inter + 1) * 4
        orig_inter_indices = list(range(n_inter_inter_edges, len(edges)))
        orig_inter_indices = self._extract_traverable_indices(indices=orig_inter_indices, edges=edges, sigs=sigs)
        orig_inter_edges = [(edges[idx][0], edges[idx][1]) for idx in orig_inter_indices]
        orig_inter_travel_time = [travel_time[i, j] for i, j in orig_inter_edges]
        # construct graph
        G = self._construct_network(edges=orig_inter_edges, attrs={'travel_time': orig_inter_travel_time})
        return self._get_reachable_shortest_length(G=G, cutoff=self.time_limit), G

    def _get_reachable_shortest_length(self, G, cutoff, path=False):
        out = {}
        for orig in range(self.n_orig):
            if path:
                reachable = single_source_dijkstra_path(G=G, source=orig, cutoff=cutoff, weight='travel_time')
            else:
                reachable = single_source_dijkstra_path_length(G=G, source=orig, cutoff=cutoff, weight='travel_time')
            reachable_des = self._extract_reachable_des(reachable=reachable, orig=orig)
            out[orig] = reachable_des
        return out

    def _extract_reachable_des(self, reachable, orig):
        """
        extract the shortest path/length from the origin to the destinations
        :param lengths: dict, reachable_node: shortest_path/length
        :return: dict, reachable_destination: shortest_path/length
        """
        reachable_des = [n for n in list(reachable.keys()) if n < self.n_orig and n != orig]
        reachable_lengths = {des: reachable[des] for des in reachable_des}
        return reachable_lengths

    @staticmethod
    def _extract_traverable_indices(indices, edges, sigs):
        """
        extract from the given edges the edges whose origin has traffic signals
        :param indices: a list of edge indices that we care about
        :param edges: a list of edges
        :param sigs: binary list representing if each node has traffic signals or not
        :return: a list of indices
        """
        return [idx for idx in indices if sigs[edges[idx][0]] == 1]

    def _construct_network(self, edges, attrs):
        edge_w_attr = self._add_edge_attr(edges, attrs)
        G = nx.DiGraph()
        G.add_edges_from(edge_w_attr)
        return G

    @staticmethod
    def _add_edge_attr(edges, attrs):
        """
        add arttribute to edges and organize it in a format that can be intaken by nx
        :param edges: a list of edges
        :param attrs: a dict of attributes, attr (key): values
        :return: [(orig, des, dict of attributes)]
        """
        edge_w_attr = []
        attr_list = list(attrs.keys())
        for idx, e in enumerate(edges):
            attr = {a: attrs[a][idx] for a in attr_list}
            edge_w_attr.append((e[0], e[1], attr))
        return edge_w_attr

    def _extract_od_pairs(self, od_overall, od_connected):
        """
        extract the od pairs that could be connected but are not connected now
        :param od_overall: the dict of the reachable destinations for each origin on the overall network
        :param od_connected: the dict of the connected destinations for each origin on the current network
        :return: dict, key: orig, value: dict of destinations
        """
        if len(od_connected) == 0:
            return od_overall
        else:
            return {orig: {des: length
                           for des, length in destinations.items()
                           if des not in od_connected[orig]}
                    for orig, destinations in od_overall.items()}

    def _cal_acc(self, destinations, pop):

        acc, cnt = 0, 0
        for orig in range(self.n_orig):
            cnt += len(destinations[orig])
            for des in destinations[orig]:
                acc += pop[des]
        return acc, cnt

    @staticmethod
    def _merge_path_dict(d1, d2):
        d = d1.copy()
        for orig in d1:
            for des in d1[orig]:
                d[orig][des].append(d2[orig][des])
        return d

    @staticmethod
    def _gen_sp_dist(G, pair_dict):
        shortest_dist = {}
        for orig, destinations in pair_dict.items():
            reachable = single_source_dijkstra_path_length(G=G, source=orig, cutoff=1e5, weight='travel_time')
            reachable = {des: reachable[des] for des in destinations}
            shortest_dist[orig] = reachable.copy()
        return shortest_dist


class RealInstanceGeneratorWoYonge:

    def __init__(self):
        pass

    def generate(self, T=30):
        """
        generate a MaxANDP instance based on Toronto's road network
        :param T: float, travel time limit in minute
        :return: dict of instance components
        """
        # path = './prob/trt/args_wo_yonge.pkl'
        # args, load_succeed = load_file(path)
        # if load_succeed:
        #     if 'job' not in args:
        #         print('loading job data')
        #         args['job'] = self._load_job()
        #         dump_file(path, args)
        #     return args
        G = self._load_network_nx()
        G = self.change_yonge_lts(G)
        G = self._add_node_feature(G)
        G_curr = self._build_current_nx(G)
        da2node = self._load_da2node()
        pop = self._load_pop(da2node)
        travel_time = self._gen_travel_time(G)
        od_pairs_all = self._gen_reachable_pairs(G, da2node, T)
        print('# of OD pairs: {}'.format(len(des2od(od_pairs_all))))
        print('# of DA nodes (DAs might be projected to the same node): {}'.format(len(od_pairs_all)))
        od_pairs_conn = self._gen_connected_pairs(G_curr, da2node, T)
        print('# of OD pairs connected: {}'.format(len(des2od(od_pairs_conn))))
        destinations = self._extract_od_pairs(od_pairs_all, od_pairs_conn)
        print('# of remained OD pairs: {}'.format(len(des2od(destinations))))
        print('# of DAs that have at least one remained destination: {}'.format(len(destinations)))
        # generate projects
        G_art = self._build_art_graph(G)
        projects, proj_costs, edge2proj, proj_ends = self._gen_projects(G_art)
        sig_cost = self._load_sig_costs(G_art)
        print('# of projects: {}'.format(len(projects)))
        print('Mean distance of each projects: {:.2f} km'.format(self._time2distance(np.mean(list(proj_costs.values())))))
        print('# of intersections on arterials w/o signals: {}'.format(len(sig_cost)))
        print('# of edges involved: {}'.format(len(edge2proj)))
        # self._dict2shp(proj_ends)
        # filter od pairs with embedding
        args = {'destinations': destinations,
                'populations': pop,
                'G': G,
                'G_curr': G_curr,
                'travel_time': travel_time,
                'travel_time_limit': T-2,
                'travel_time_max': T,
                'projects': projects,
                'project_costs': proj_costs,
                'signal_costs': sig_cost,
                'edge2proj': edge2proj}
        # dump_file('./prob/trt/args_wo_yonge.pkl', args)
        # args['destinations'] = self._filter_od(destinations)
        # dump_file(path, args)
        return args

    @staticmethod
    def change_yonge_lts(G):
        yonge = [(13463436, 13463281), (13463281, 13463257), (13463257, 13463167), (13463167, 13463136),
                 (13463136, 13463047), (13463047, 13463015), (13463015, 13462676), (13462676, 13462611),
                 (13462611, 13462557), (13462557, 14014263), (14014263, 13462438), (13462438, 30078034),
                 (30078034, 13462319), (13462319, 13462305), (13462305, 13462255), (13462255, 13462159),
                 (13462159, 13462084), (13462084, 13461921), (13461921, 13461783), (13461783, 13973615),
                 (13973615, 13461519), (13461519, 13461419), (13461419, 13461390), (13461390, 13461257),
                 (13461257, 13461211), (13461211, 13461050), (13461050, 13460866), (13460866, 13460709),
                 (13460709, 13460639), (13460639, 13460473), (13460473, 13460295), (13460295, 13460106),
                 (13460106, 14242420), (14242420, 13459944), (13459944, 13459913), (13459913, 13459811),
                 (13459811, 13459743), (13459743, 30086603), (30086603, 13458904), (13458904, 13458807),
                 (13458807, 13458578), (13458578, 13458407)]
        for fnode, tnode in yonge:
            G[fnode][tnode]['lts'] = 4
            G[tnode][fnode]['lts'] = 4
        return G

    def _gen_reachable_pairs(self, G, da2node, T):
        path = './data/trt_instance/od_pairs_all.pkl'
        od_pairs, load_succeed = load_file(path)
        if load_succeed:
            return od_pairs
        od_pairs = {}
        destinations = set(list(da2node.values()))
        for da, node in tqdm(da2node.items()):
            reachable = self._orig2des_length(node, destinations, T, G)
            if len(reachable) > 0:
                od_pairs[node] = reachable
        dump_file(path, od_pairs)
        return od_pairs

    def _gen_connected_pairs(self, G, da2node, T):
        od_pairs = {}
        destinations = set(list(da2node.values()))
        for da, node in tqdm(da2node.items()):
            reachable = self._orig2des_length(node, destinations, T, G)
            if len(reachable) > 0:
                od_pairs[node] = reachable
        dump_file('./prob/trt/connected_wo_yonge.pkl', od_pairs)
        return od_pairs

    def _gen_projects(self, G):
        # path_proj = './data/trt_instance/projects_wo.pkl'
        # path_projcost = './data/trt_instance/project_costs_wo.pkl'
        # path_edge2proj = './data/trt_instance/edge2projs_wo.pkl'
        # path_projends = './data/trt_instance/projends_wo.pkl'
        # projects, load1 = load_file(path_proj)
        # proj_costs, load2 = load_file(path_projcost)
        # edge2proj, load3 = load_file(path_edge2proj)
        # proj_ends, load4 = load_file(path_projends)
        # if load1 and load2 and load3 and load4:
        #     return projects, proj_costs, edge2proj, proj_ends
        # find stopping nodes
        path_artinter = './data/trt_arterial/art_inter.pkl'
        art_inter, _ = load_file(path_artinter)
        deadends = self._find_deadends(G)
        stopping_nodes = art_inter.copy()
        stopping_nodes.update(deadends)
        # manually add some nodes based on inspection
        stopping_nodes.update([13450841, 13445621, 13449375, 13459662, 13462952, 30015319])
        # manually remove some arterial nodes connected w/ ramps
        stopping_nodes.difference_update([13461547, 13461559, 13467693, 13467543, 13465651, 13465666, 13453408, 13445579,
                                          13465743, 13465776, 13455544, 13457591, 13455558, 13451486, 13455588, 30017765,
                                          13455618, 13467911, 13455628, 13467920, 13469980, 13470018, 13461841, 13451615,
                                          13455715, 13455739, 13451668, 13451682, 13468086, 13461948, 13463999, 13459908,
                                          13459908, 13451735, 13470174, 13470179, 13468140, 13451763, 13455872, 13468168,
                                          13451798, 13451809, 13453865, 13460013, 13453881, 13468222, 13460041, 13453898,
                                          13462097, 30005847, 30003677, 30003616])
        # remove nodes on ramps
        container = stopping_nodes.copy()
        for node in container:
            if node not in G.nodes():
                stopping_nodes.remove(node)
        # generate projects
        edge_existance = {(fnode, tnode) for fnode, tnode in G.edges()}
        projects = []
        proj_costs = {}
        edge2proj = {}
        proj_ends = []
        idx = -1
        for node in stopping_nodes:
            new_project, traversed_edges = self._find_project(node, G, stopping_nodes, edge_existance)
            G = self._update_art_graph(G, traversed_edges)
            for fnode, tnode, cost, edges in new_project:
                if cost > 0:
                    idx += 1
                    proj_ends.append((fnode, tnode))
                    projects.append(edges)
                    proj_costs[idx] = cost
                    for i, j in edges:
                        edge2proj[(i, j)] = idx
        # store the files
        # dump_file(path_proj, projects)
        # dump_file(path_projcost, proj_costs)
        # dump_file(path_edge2proj, edge2proj)
        # dump_file(path_projends, proj_ends)
        return projects, proj_costs, edge2proj, proj_ends

    def _find_project(self, node, G, stopping_node, edge_existance):
        projects = []
        traversed_edges = []
        for fnode, tnode in G.out_edges(node):
            new_projects, traversed = self._search_forward(fnode, tnode, G, stopping_node, edge_existance)
            projects.append(new_projects)
            traversed_edges += traversed
        return projects, traversed_edges

    def _search_forward(self, fnode, tnode, G, stopping_node, edge_existance):
        start_node = fnode
        length, edges, traversed_edges = self._add_edge_to_proj(G, fnode, tnode, edge_existance)
        while tnode not in stopping_node:
            next_nodes = self._possible_next_nodes(G, fnode, tnode)
            if len(next_nodes) != 1:
                raise ValueError('direction {} -> {} should have only one next nodes'.format(fnode, tnode),
                                 [(tnode, node) for node in next_nodes])
            fnode, tnode = tnode, next_nodes[0]
            add_length, add_edges, add_t_edges = self._add_edge_to_proj(G, fnode, tnode, edge_existance)
            edges += add_edges
            length += add_length
            traversed_edges += add_t_edges
        new_project = (start_node, tnode, length, edges)
        return new_project, traversed_edges

    @staticmethod
    def _possible_next_nodes(G, fnode, tnode):
        return [e2 for e1, e2 in G.out_edges(tnode) if e2 != fnode]

    @staticmethod
    def _add_edge_to_proj(G, fnode, tnode, edge_existance):
        length = 0
        add_edges = []
        traversed_edges = []
        if G[fnode][tnode]['lts'] > 2:
            length += G[fnode][tnode]['time']
            add_edges.append((fnode, tnode))
        if ((tnode, fnode) in edge_existance) and (G[tnode][fnode]['lts'] > 2):
            length += G[tnode][fnode]['time']
            add_edges.append((tnode, fnode))
        return length, add_edges, [(e1, e2) for e1, e2 in [(fnode, tnode), (tnode, fnode)] if (e1, e2) in edge_existance]

    @staticmethod
    def _update_art_graph(G, traversed_edges):
        for fnode, tnode in traversed_edges:
            G.remove_edge(fnode, tnode)
        return G

    def _find_deadends(self, G):
        deadends = set()
        for node in G.nodes():
            if self._deadend(G, node):
                deadends.add(node)
        return deadends

    @staticmethod
    def _deadend(G, node):
        conn_nodes = set([tnode for fnode, tnode in G.out_edges(node)] + [fnode for fnode, tnode in G.in_edges(node)])
        return len(conn_nodes) <= 1

    @staticmethod
    def _build_art_graph(G):
        # path = './data/trt_arterial/G_art.pkl'
        # G_art, load_succeed = load_file(path)
        # if load_succeed:
        #     return G_art
        art_node, _ = load_file('./data/trt_arterial/artnode.pkl')
        edges = []
        restricted_edges = {(13442443, 13442540), (13442693, 13442850), (13442992, 13443301),
                            (13445303, 13444638), (13445264, 13444687), (30000717, 30000714),
                            (13469764, 13469954), (30033283, 30033294), (13446990, 13446389),
                            (13446915, 13446334), (13445912, 13446002), (30020937, 30020931),
                            (13465719, 13466000), (13465741, 13466016), (13465028, 13464880),
                            (13467597, 13467854), (13467532, 13467771), (13467665, 13467447),
                            (13467398, 13467612), (13467447, 13467437), (13465590, 13465699),
                            (13467695, 13467525), (13465816, 13465526), (13459699, 13459658),
                            (14228400, 14020768), (13445579, 13445678), (13445752, 13445392),
                            (13459322, 13459590), (14073966, 13975054), (13459710, 13459781),
                            (20154546, 14659268), (14624109, 30090087), (30008693, 30008688),
                            (13459330, 30067029), (14673438, 14673411), (13446475, 13446617),
                            (13468264, 13468369), (13463318, 13463450), (20006075, 20006080),
                            (13447296, 30020794), (20142358, 20142354), (13463181, 13463291),
                            (13465922, 13465764), (30108537, 30108540), (13447951, 13448440),
                            (13448921, 13448760), (13448921, 13449203), (13448484, 13448352),
                            (13448098, 13447994), (13465878, 13465784), (13465878, 13465897),
                            (13455060, 13454586), (13454642, 13453989), (13455348, 13455332),
                            (13466378, 13466441), (13462426, 13462399), (30020057, 30000570),
                            (13463876, 20089401), (13463445, 13463707), (13463334, 13463306),
                            (13465932, 13465652), (13465932, 13466175), (13465922, 13466039),
                            (13450723, 13451142), (30020928, 30020931), (13448840, 13448916),
                            (13466159, 13466039), (13449519, 13449359), (13463895, 13464456),
                            (13463660, 13463756), (13463660, 13463511), (13463075, 13462958),
                            (13466119, 13465975), (13466279, 13466123), (13465864, 13465817),
                            (13445948, 13446020), (13468184, 13468549), (13461841, 30003677),
                            (13466000, 13466243), (13466034, 13466287), (13457922, 13457551),
                            (13457060, 13456668), (13457060, 13457328), (30000682, 30000676),
                            (13466020, 13466205), (30003732, 30003741), (30037447, 13468065),
                            (13456704, 13456778), (13464006, 13463588), (14023828, 14023782),
                            (13464125, 13464408), (13449274, 13449578), (13462034, 20145559),
                            (30003622, 30003616), (20229269, 13456777), (13457588, 13457586),
                            (13452501, 13452496), (13503859, 13444544), (13454932, 13454935),
                            (30001193, 30001196), (13451353, 20229251), (13450806, 20229260),
                            (20362228, 20230204), (13442710, 13442417), (13468079, 13468195),
                            (14134984, 14134885), (13468195, 13468256), (13459724, 13460109),
                            (30003689, 30057571), (14134864, 14048060), (14238996, 13452098),
                            (13444159, 13444344), (30002250, 30002247), (13459076, 13458802),
                            (13447710, 13447784), (13447414, 13446751), (13466254, 13466381),
                            (13458803, 13459527), (13468316, 13468256), (13468466, 13468377),
                            (14130687, 13973648), (13466249, 13466440), (14256135, 13442074),
                            (13460471, 13459524), (13468319, 13468532), (13452531, 13451895),
                            (13454909, 13454774), (13456246, 13456149), (30038667, 20364357),
                            (13441641, 13441651), (13441659, 13441760), (13466287, 13466399),
                            (13460340, 13458634), (30003505, 30003499), (13446221, 13446407),
                            (13466333, 13466475), (13466333, 13466205), (30011656, 13469031),
                            (13454074, 13453844), (20233584, 13973648), (13466582, 13466382),
                            (13463924, 13463834), (13454867, 13454486), (30110360, 30000992),
                            (13456161, 13455616), (13468360, 13468529), (13468559, 13468687),
                            (13468605, 13468807), (13452794, 13452757), (30004915, 30004906),
                            (13455765, 13456371), (13466266, 13466404), (13450347, 13450291),
                            (13463608, 13463438), (13455929, 13456196), (14174334, 14174328),
                            (13460887, 13460553), (13464635, 14011907), (30007497, 30007494),
                            (13468529, 13468553), (13468687, 13468712), (13468807, 13468837),
                            (30006406, 30006409), (13466551, 13466636), (13464714, 13464755),
                            (13466440, 13466590), (13466389, 13466341), (13466389, 13466537),
                            (13467037, 13467132), (13468632, 13468799), (13466537, 13466685),
                            (13466590, 13466733), (13456778, 13456962), (13464439, 13464852),
                            (14659662, 30006051), (13454346, 13454651), (14129762, 13446725),
                            (13447797, 13447873), (13447920, 13448020), (13448061, 13448124),
                            (13448225, 13448315), (13450698, 13450944), (13455212, 13455307),
                            (13455392, 13455471), (13455675, 13455752), (13455827, 13455890),
                            (13455968, 13456039), (13456084, 13456148), (13456416, 13456476),
                            (13456650, 13456723), (13456727, 13456803), (13456863, 13456935),
                            (30062739, 13458632), (13446504, 20233934), (14067768, 14067820), (13448244, 13448355),
                            (13973970, 13973964), (30094088, 30094091), (13455608, 13455700),
                            (13462709, 13462699), (30003499, 30003493), (13446506, 13446407),
                            (13467006, 13467144), (13446407, 14249902), (13458957, 13458985), (30002743, 30002750),
                            (30005843, 30005847), (30005853, 30005847), (13449594, 13449802), (20233509, 30013028),
                            (13462900, 30015319), (13452777, 13453021), (13465052, 13465422), (14014179, 13463460),
                            (13467128, 13467278), (13467433, 13467265), (30044055, 30044052), (13463136, 13463087),
                            (13448802, 13448861), (13467173, 13467394), (13467242, 13467452), (14025537, 14025545),
                            (13467277, 13467481), (13444915, 13444898), (30002746, 30002757), (20102847, 13467265),
                            (13469326, 13469273), (14254936, 14254942), (14011907, 13464509), (13467373, 13467146),
                            (13467394, 13467593), (13465390, 13465773), (13447113, 13447074), (13465164, 13465121),
                            (13469439, 13469409), (13461841, 30003689), (30003677, 13463374), (13462945, 13463078),
                            (13462959, 13463068), (30071298, 30015319), (30071298, 13463374)}
        restricted_pool = restricted_edges.copy()
        for (fnode, tnode) in restricted_edges:
            restricted_pool.add((tnode, fnode))
        for fnode, tnode in G.edges():
            if (fnode, tnode) in restricted_pool:
                continue
            if (fnode in art_node) and (tnode in art_node):
                edges.append((fnode, tnode, G[fnode][tnode]))
        G_art = nx.DiGraph()
        G_art.add_edges_from(edges)
        # dump_file(path, G_art)
        return G_art

    @staticmethod
    def _gen_compression_dict(G, da2node):
        path_node2idx = './data/trt_instance/node2idx.pkl'
        path_idx2node = './data/trt_instance/idx2node.pkl'
        node2idx, succeed_1 = load_file(path_node2idx)
        idx2node, succeed_2 = load_file(path_idx2node)
        if succeed_1 and succeed_2:
            return node2idx, idx2node
        node2idx, idx2node = {}, {}
        idx = -1
        # label the DA nodes:
        for da, node in da2node.items():
            if node not in G.nodes():
                continue
            idx += 1
            node2idx[node] = idx
            idx2node[idx] = node
        # label the rest
        for node in G.nodes():
            if node in node2idx:
                continue
            idx += 1
            node2idx[node] = idx
            idx2node[idx] = node
        dump_file(path_node2idx, node2idx)
        dump_file(path_idx2node, idx2node)
        return node2idx, idx2node

    @staticmethod
    def _graph_compression(G, node2idx, name):
        path = './data/trt_instance/{}_comp.pkl'.format(name)
        G_new, load_succeed = load_file(path)
        if load_succeed:
            return G_new
        edges = []
        for fnode, tnode in G.edges():
            edge = (node2idx[fnode], node2idx[tnode], G[fnode][tnode])
            edges.append(edge)
        G_new = nx.DiGraph()
        G_new.add_edges_from(edges)
        dump_file(path, G_new)
        return G_new

    @staticmethod
    def _od_compression(od_pairs, node2idx, name):
        path = './data/trt_instance/od_pairs_{}_comp.pkl'.format(name)
        od_comp, load_succeed = load_file(path)
        if load_succeed:
            return od_comp
        od_comp = {node2idx[orig]: {node2idx[des]: od_pairs[orig][des] for des in od_pairs[orig]} for orig in od_pairs}
        dump_file(path, od_comp)
        return od_comp

    @staticmethod
    def _extract_od_pairs(od_all, od_conn):
        od_remained = {orig: {des: length for des, length in destination.items()
                               if (orig not in od_conn) or ((orig in od_conn) and (des not in od_conn[orig]))}
                       for orig, destination in od_all.items()}
        od_remained = {orig: destination for orig, destination in od_remained.items() if len(destination) > 0}
        return od_remained

    @staticmethod
    def _build_current_nx(G):
        edges = []
        fake_intersect = set(list(pd.read_csv('./data/trt_sig_intersect/fake_intersections.csv')['FNODE'].values))
        for fnode, tnode in G.edges():
            if ((G[fnode][tnode]['lts'] > 2) or (G.nodes[fnode]['lts'] > 2)) and (fnode not in fake_intersect):
                cost = 100
            else:
                cost = G[fnode][tnode]['time']
            edges.append((fnode, tnode, {'time': cost}))
        G_curr = nx.DiGraph()
        G_curr.add_edges_from(edges)
        return G_curr

    @staticmethod
    def _add_node_feature(G):
        node2sig, _ = load_file('./data/trt_sig_intersect/node2sig.pkl')
        attrs = {}
        for node in G.nodes:
            lts = 2
            if not node2sig[node]:
                for _, tnode in G.out_edges(node):
                    if G[node][tnode]['lts'] >= 3:
                        lts = 3
                        break
                for fnode, _ in G.in_edges(node):
                    if G[fnode][node]['lts'] >= 3:
                        lts = 3
                        break
            attrs[node] = lts
        nx.set_node_attributes(G, attrs, 'lts')
        return G

    @staticmethod
    def _orig2des_length(orig, destinations, T, G):
        if orig not in G.nodes():
            return {}
        reachable = single_source_dijkstra_path_length(G=G, source=orig, cutoff=T, weight='time')
        return {des: val for des, val in reachable.items() if des in destinations and des != orig}

    @staticmethod
    def _load_network_nx():
        path = './data/trt_network/final/G_simplified.pkl'
        G, _ = load_file(path)
        return G

    @staticmethod
    def _load_da2node():
        path_da2node = './data/trt_da/da2node.pkl'
        da2node, _ = load_file(path_da2node)
        return da2node

    @staticmethod
    def _load_sig_costs(G):
        path = './data/trt_sig_intersect/node2sig.pkl'
        node2sig, _ = load_file(path)
        fake_intersect = set(list(pd.read_csv('./data/trt_sig_intersect/fake_intersections.csv')['FNODE'].values))
        return {node: 1 for node in G.nodes() if not node2sig[node] and node not in fake_intersect}

    @staticmethod
    def _load_pop(da2node):
        path_da = './data/trt_da/da.pkl'
        df_da, _ = load_file(path_da)
        dauid = df_da['DAUID'].values
        pop = df_da['pop'].values
        node_pop = {da2node[da]: 0 for da in dauid}
        for idx in range(len(pop)):
            node_pop[da2node[dauid[idx]]] += pop[idx]
        return node_pop

    @staticmethod
    def _load_job():
        path_da2node = './data/trt_da/da2node.pkl'
        da2node, _ = load_file(path_da2node)
        df_da = pd.read_excel('./data/trt_da/pop_job_census.xls')
        dauid = df_da['DAUID'].values
        job = df_da['Census_job'].values
        node_job = {da2node[str(da)]: 0 for da in dauid}
        for idx in range(len(job)):
            node_job[da2node[str(dauid[idx])]] += job[idx]
        return node_job

    @staticmethod
    def _load_fake_intersection():
        path = './data/trt_sig_intersect/fake_intersections.csv'
        df = pd.read_csv(path)
        return set(list(df['FNODE'].values))

    @staticmethod
    def _gen_travel_time(G):
        return {(fnode, tnode): G[fnode][tnode]['time'] for fnode, tnode in G.edges()}

    # def _dict2shp(self, proj_ends):
    #     print('Generating a shapefile for the projects')
    #     path_node2loc = './data/trt_network/intermediate/node_loc.pkl'
    #     node2loc, _ = load_file(path_node2loc)
    #     tnode, fnode, length, time, geometry = [], [], [], [], []
    #     for f, t in proj_ends:
    #         tnode.append(t)
    #         fnode.append(f)
    #         time.append(proj_ends[(f, t)])
    #         length.append(self._time2distance(proj_ends[(f, t)]))
    #         geometry.append(LineString([node2loc[f], node2loc[t]]))
    #     df = pd.DataFrame({'fnode': fnode, 'tnode': tnode, 'time': time, 'length': length})
    #     gdf = gpd.GeoDataFrame(df, geometry=geometry)
    #     gdf.to_file(driver='ESRI Shapefile', filename='./data/trt_instance/project_shapefile/proj.shp')

    @staticmethod
    def _time2distance(t):
        return t * 15000 / 60

    @staticmethod
    def _distance2time(d):
        return d * 60 / 15000

    @staticmethod
    def _filter_od(destination):
        emb_pairs, _ = load_file('./prob/trt/emb/emb_pairs32.pkl')
        feature, _ = load_file('./prob/trt/emb/feature32.pkl')
        args_old, _ = load_file('./prob/trt/arg_archive_wo_Yonge/args_wo_yonge.pkl')
        # remove od pairs w/o emb
        od_pairs_w_emb = [(orig, des) for orig in args_old['destinations'] for des in args_old['destinations'][orig]]
        od_pairs_w_emb = np.array(od_pairs_w_emb)[emb_pairs]
        destination_new = {}
        feature_idx = {}
        cnt = -1
        for orig, des in od_pairs_w_emb:
            cnt += 1
            if (orig not in destination) or (des not in destination[orig]):
                continue
            if orig in destination_new:
                destination_new[orig].update({des: 1})
                feature_idx[orig].update({des: cnt})
            else:
                destination_new[orig] = {des: 1}
                feature_idx[orig] = {des: cnt}
        feature_new = [feature[feature_idx[orig][des]] for orig in destination_new for des in destination_new[orig]]
        feature_new = np.array(feature_new)
        dump_file('./prob/trt/emb/final/emb.pkl', feature_new)
        return destination_new


class RealInstanceGenerator:

    def __init__(self):
        pass

    def generate(self, region, T=30):
        """
        generate a MaxANDP instance based on Toronto's road network
        :param T: float, travel time limit in minute
        :return: dict of instance components
        """
        path = './prob/trt/args_adj_ratio_%s.pkl' %region
        args, load_succeed = load_file(path)
        if load_succeed:
            if 'job' not in args:
                print('loading job data')
                args['job'] = self._load_job()
                dump_file(path, args)
            return args
        G = self._load_network_nx()
        G = self._add_node_feature(G)
        G_curr = self._build_current_nx(G)
        da2node = self._load_da2node()
        pop = self._load_pop(da2node)
        travel_time = self._gen_travel_time(G)
        od_pairs_all = self._gen_reachable_pairs(G, da2node, T)
        print('# of OD pairs: {}'.format(len(des2od(od_pairs_all))))
        print('# of DA nodes (DAs might be projected to the same node): {}'.format(len(od_pairs_all)))
        od_pairs_conn = self._gen_connected_pairs(G_curr, da2node, T)
        print('# of OD pairs connected: {}'.format(len(des2od(od_pairs_conn))))
        destinations = self._extract_od_pairs(od_pairs_all, od_pairs_conn)
        print('# of remained OD pairs: {}'.format(len(des2od(destinations))))
        print('# of DAs that have at least one remained destination: {}'.format(len(destinations)))
        # generate projects
        G_art = self._build_art_graph(G)
        projects, proj_costs, edge2proj, proj_ends = self._gen_projects(G_art)
        sig_cost = self._load_sig_costs(G_art)
        print('# of projects: {}'.format(len(projects)))
        print('Mean distance of each projects: {:.2f} km'.format(self._time2distance(np.mean(list(proj_costs.values())))))
        print('# of intersections on arterials w/o signals: {}'.format(len(sig_cost)))
        print('# of edges involved: {}'.format(len(edge2proj)))
        # self._dict2shp(proj_ends)
        # filter od pairs with embedding
        args = {'destinations': destinations,
                'populations': pop,
                'G': G,
                'G_curr': G_curr,
                'travel_time': travel_time,
                'travel_time_limit': T-2,
                'travel_time_max': T,
                'projects': projects,
                'project_costs': proj_costs,
                'signal_costs': sig_cost,
                'edge2proj': edge2proj}
        dump_file('./prob/trt/args.pkl', args)
        args['destinations'] = self._filter_od(destinations)
        dump_file(path, args)
        return args

    def _gen_reachable_pairs(self, G, da2node, T):
        path = './data/trt_instance/od_pairs_all.pkl'
        od_pairs, load_succeed = load_file(path)
        if load_succeed:
            return od_pairs
        od_pairs = {}
        destinations = set(list(da2node.values()))
        for da, node in tqdm(da2node.items()):
            reachable = self._orig2des_length(node, destinations, T, G)
            if len(reachable) > 0:
                od_pairs[node] = reachable
        dump_file(path, od_pairs)
        return od_pairs

    def _gen_connected_pairs(self, G, da2node, T):
        path = './data/trt_instance/od_pairs_conn.pkl'
        od_pairs, load_succeed = load_file(path)
        if load_succeed:
            return od_pairs
        od_pairs = {}
        destinations = set(list(da2node.values()))
        for da, node in tqdm(da2node.items()):
            reachable = self._orig2des_length(node, destinations, T, G)
            if len(reachable) > 0:
                od_pairs[node] = reachable
        dump_file(path, od_pairs)
        return od_pairs

    def _gen_projects(self, G):
        path_proj = './data/trt_instance/projects.pkl'
        path_projcost = './data/trt_instance/project_costs.pkl'
        path_edge2proj = './data/trt_instance/edge2projs.pkl'
        path_projends = './data/trt_instance/projends.pkl'
        projects, load1 = load_file(path_proj)
        proj_costs, load2 = load_file(path_projcost)
        edge2proj, load3 = load_file(path_edge2proj)
        proj_ends, load4 = load_file(path_projends)
        if load1 and load2 and load3 and load4:
            return projects, proj_costs, edge2proj, proj_ends
        # find stopping nodes
        path_artinter = './data/trt_arterial/art_inter.pkl'
        art_inter, _ = load_file(path_artinter)
        deadends = self._find_deadends(G)
        stopping_nodes = art_inter.copy()
        stopping_nodes.update(deadends)
        # manually add some nodes based on inspection
        stopping_nodes.update([13450841, 13445621, 13449375, 13459662, 13462952])
        # manually remove some arterial nodes connected w/ ramps
        stopping_nodes.difference_update([13461547, 13461559, 13467693, 13467543, 13465651, 13465666, 13453408, 13445579,
                                          13465743, 13465776, 13455544, 13457591, 13455558, 13451486, 13455588, 30017765,
                                          13455618, 13467911, 13455628, 13467920, 13469980, 13470018, 13461841, 13451615,
                                          13455715, 13455739, 13451668, 13451682, 13468086, 13461948, 13463999, 13459908,
                                          13459908, 13451735, 13470174, 13470179, 13468140, 13451763, 13455872, 13468168,
                                          13451798, 13451809, 13453865, 13460013, 13453881, 13468222, 13460041, 13453898,
                                          13462097, 30005847])
        # remove nodes on ramps
        container = stopping_nodes.copy()
        for node in container:
            if node not in G.nodes():
                stopping_nodes.remove(node)
        # generate projects
        edge_existance = {(fnode, tnode) for fnode, tnode in G.edges()}
        projects = []
        proj_costs = {}
        edge2proj = {}
        proj_ends = []
        idx = -1
        for node in stopping_nodes:
            new_project, traversed_edges = self._find_project(node, G, stopping_nodes, edge_existance)
            G = self._update_art_graph(G, traversed_edges)
            for fnode, tnode, cost, edges in new_project:
                if cost > 0:
                    idx += 1
                    proj_ends.append((fnode, tnode))
                    projects.append(edges)
                    proj_costs[idx] = cost
                    for i, j in edges:
                        edge2proj[(i, j)] = idx
        # store the files
        dump_file(path_proj, projects)
        dump_file(path_projcost, proj_costs)
        dump_file(path_edge2proj, edge2proj)
        dump_file(path_projends, proj_ends)
        return projects, proj_costs, edge2proj, proj_ends

    def _find_project(self, node, G, stopping_node, edge_existance):
        projects = []
        traversed_edges = []
        for fnode, tnode in G.out_edges(node):
            new_projects, traversed = self._search_forward(fnode, tnode, G, stopping_node, edge_existance)
            projects.append(new_projects)
            traversed_edges += traversed
        return projects, traversed_edges

    def _search_forward(self, fnode, tnode, G, stopping_node, edge_existance):
        start_node = fnode
        length, edges, traversed_edges = self._add_edge_to_proj(G, fnode, tnode, edge_existance)
        while tnode not in stopping_node:
            next_nodes = self._possible_next_nodes(G, fnode, tnode)
            if len(next_nodes) != 1:
                raise ValueError('direction {} -> {} should have only one next nodes'.format(fnode, tnode),
                                 [(tnode, node) for node in next_nodes])
            fnode, tnode = tnode, next_nodes[0]
            add_length, add_edges, add_t_edges = self._add_edge_to_proj(G, fnode, tnode, edge_existance)
            edges += add_edges
            length += add_length
            traversed_edges += add_t_edges
        new_project = (start_node, tnode, length, edges)
        return new_project, traversed_edges

    @staticmethod
    def _possible_next_nodes(G, fnode, tnode):
        return [e2 for e1, e2 in G.out_edges(tnode) if e2 != fnode]

    @staticmethod
    def _add_edge_to_proj(G, fnode, tnode, edge_existance):
        length = 0
        add_edges = []
        traversed_edges = []
        if G[fnode][tnode]['lts'] > 2:
            length += G[fnode][tnode]['time']
            add_edges.append((fnode, tnode))
        if ((tnode, fnode) in edge_existance) and (G[tnode][fnode]['lts'] > 2):
            length += G[tnode][fnode]['time']
            add_edges.append((tnode, fnode))
        return length, add_edges, [(e1, e2) for e1, e2 in [(fnode, tnode), (tnode, fnode)] if (e1, e2) in edge_existance]

    @staticmethod
    def _update_art_graph(G, traversed_edges):
        for fnode, tnode in traversed_edges:
            G.remove_edge(fnode, tnode)
        return G

    def _find_deadends(self, G):
        deadends = set()
        for node in G.nodes():
            if self._deadend(G, node):
                deadends.add(node)
        return deadends

    @staticmethod
    def _deadend(G, node):
        conn_nodes = set([tnode for fnode, tnode in G.out_edges(node)] + [fnode for fnode, tnode in G.in_edges(node)])
        return len(conn_nodes) <= 1

    @staticmethod
    def _build_art_graph(G):
        path = './data/trt_arterial/G_art.pkl'
        G_art, load_succeed = load_file(path)
        if load_succeed:
            return G_art
        art_node, _ = load_file('./data/trt_arterial/artnode.pkl')
        edges = []
        restricted_edges = {(13442443, 13442540), (13442693, 13442850), (13442992, 13443301),
                            (13445303, 13444638), (13445264, 13444687), (30000717, 30000714),
                            (13469764, 13469954), (30033283, 30033294), (13446990, 13446389),
                            (13446915, 13446334), (13445912, 13446002), (30020937, 30020931),
                            (13465719, 13466000), (13465741, 13466016), (13465028, 13464880),
                            (13467597, 13467854), (13467532, 13467771), (13467665, 13467447),
                            (13467398, 13467612), (13467447, 13467437), (13465590, 13465699),
                            (13467695, 13467525), (13465816, 13465526), (13459699, 13459658),
                            (14228400, 14020768), (13445579, 13445678), (13445752, 13445392),
                            (13459322, 13459590), (14073966, 13975054), (13459710, 13459781),
                            (20154546, 14659268), (14624109, 30090087), (30008693, 30008688),
                            (13459330, 30067029), (14673438, 14673411), (13446475, 13446617),
                            (13468264, 13468369), (13463318, 13463450), (20006075, 20006080),
                            (13447296, 30020794), (20142358, 20142354), (13463181, 13463291),
                            (13465922, 13465764), (30108537, 30108540), (13447951, 13448440),
                            (13448921, 13448760), (13448921, 13449203), (13448484, 13448352),
                            (13448098, 13447994), (13465878, 13465784), (13465878, 13465897),
                            (13455060, 13454586), (13454642, 13453989), (13455348, 13455332),
                            (13466378, 13466441), (13462426, 13462399), (30020057, 30000570),
                            (13463876, 20089401), (13463445, 13463707), (13463334, 13463306),
                            (13465932, 13465652), (13465932, 13466175), (13465922, 13466039),
                            (13450723, 13451142), (30020928, 30020931), (13448840, 13448916),
                            (13466159, 13466039), (13449519, 13449359), (13463895, 13464456),
                            (13463660, 13463756), (13463660, 13463511), (13463075, 13462958),
                            (13466119, 13465975), (13466279, 13466123), (13465864, 13465817),
                            (13445948, 13446020), (13468184, 13468549), (13461841, 30003677),
                            (13466000, 13466243), (13466034, 13466287), (13457922, 13457551),
                            (13457060, 13456668), (13457060, 13457328), (30000682, 30000676),
                            (13466020, 13466205), (30003732, 30003741), (30037447, 13468065),
                            (13456704, 13456778), (13464006, 13463588), (14023828, 14023782),
                            (13464125, 13464408), (13449274, 13449578), (13462034, 20145559),
                            (30003622, 30003616), (20229269, 13456777), (13457588, 13457586),
                            (13452501, 13452496), (13503859, 13444544), (13454932, 13454935),
                            (30001193, 30001196), (13451353, 20229251), (13450806, 20229260),
                            (20362228, 20230204), (13442710, 13442417), (13468079, 13468195),
                            (14134984, 14134885), (13468195, 13468256), (13459724, 13460109),
                            (30003689, 30057571), (14134864, 14048060), (14238996, 13452098),
                            (13444159, 13444344), (30002250, 30002247), (13459076, 13458802),
                            (13447710, 13447784), (13447414, 13446751), (13466254, 13466381),
                            (13458803, 13459527), (13468316, 13468256), (13468466, 13468377),
                            (14130687, 13973648), (13466249, 13466440), (14256135, 13442074),
                            (13460471, 13459524), (13468319, 13468532), (13452531, 13451895),
                            (13454909, 13454774), (13456246, 13456149), (30038667, 20364357),
                            (13441641, 13441651), (13441659, 13441760), (13466287, 13466399),
                            (13460340, 13458634), (30003505, 30003499), (13446221, 13446407),
                            (13466333, 13466475), (13466333, 13466205), (30011656, 13469031),
                            (13454074, 13453844), (20233584, 13973648), (13466582, 13466382),
                            (13463924, 13463834), (13454867, 13454486), (30110360, 30000992),
                            (13456161, 13455616), (13468360, 13468529), (13468559, 13468687),
                            (13468605, 13468807), (13452794, 13452757), (30004915, 30004906),
                            (13455765, 13456371), (13466266, 13466404), (13450347, 13450291),
                            (13463608, 13463438), (13455929, 13456196), (14174334, 14174328),
                            (13460887, 13460553), (13464635, 14011907), (30007497, 30007494),
                            (13468529, 13468553), (13468687, 13468712), (13468807, 13468837),
                            (30006406, 30006409), (13466551, 13466636), (13464714, 13464755),
                            (13466440, 13466590), (13466389, 13466341), (13466389, 13466537),
                            (13467037, 13467132), (13468632, 13468799), (13466537, 13466685),
                            (13466590, 13466733), (13456778, 13456962), (13464439, 13464852),
                            (14659662, 30006051), (13454346, 13454651), (14129762, 13446725),
                            (13447797, 13447873), (13447920, 13448020), (13448061, 13448124),
                            (13448225, 13448315), (13450698, 13450944), (13455212, 13455307),
                            (13455392, 13455471), (13455675, 13455752), (13455827, 13455890),
                            (13455968, 13456039), (13456084, 13456148), (13456416, 13456476),
                            (13456650, 13456723), (13456727, 13456803), (13456863, 13456935),
                            (30062739, 13458632), (13446504, 20233934), (14067768, 14067820), (13448244, 13448355),
                            (13973970, 13973964), (30094088, 30094091), (13455608, 13455700),
                            (13462709, 13462699), (30003499, 30003493), (13446506, 13446407),
                            (13467006, 13467144), (13446407, 14249902), (13458957, 13458985), (30002743, 30002750),
                            (30005843, 30005847), (30005853, 30005847), (13449594, 13449802), (20233509, 30013028),
                            (13462900, 30015319), (13452777, 13453021), (13465052, 13465422), (14014179, 13463460),
                            (13467128, 13467278), (13467433, 13467265), (30044055, 30044052), (13463136, 13463087),
                            (13448802, 13448861), (13467173, 13467394), (13467242, 13467452), (14025537, 14025545),
                            (13467277, 13467481), (13444915, 13444898), (30002746, 30002757), (20102847, 13467265),
                            (13469326, 13469273), (14254936, 14254942), (14011907, 13464509), (13467373, 13467146),
                            (13467394, 13467593), (13465390, 13465773), (13447113, 13447074), (13465164, 13465121),
                            (13469439, 13469409)}
        restricted_pool = restricted_edges.copy()
        for (fnode, tnode) in restricted_edges:
            restricted_pool.add((tnode, fnode))
        for fnode, tnode in G.edges():
            if (fnode, tnode) in restricted_pool:
                continue
            if (fnode in art_node) and (tnode in art_node):
                edges.append((fnode, tnode, G[fnode][tnode]))
        G_art = nx.DiGraph()
        G_art.add_edges_from(edges)
        dump_file(path, G_art)
        return G_art

    @staticmethod
    def _gen_compression_dict(G, da2node):
        path_node2idx = './data/trt_instance/node2idx.pkl'
        path_idx2node = './data/trt_instance/idx2node.pkl'
        node2idx, succeed_1 = load_file(path_node2idx)
        idx2node, succeed_2 = load_file(path_idx2node)
        if succeed_1 and succeed_2:
            return node2idx, idx2node
        node2idx, idx2node = {}, {}
        idx = -1
        # label the DA nodes:
        for da, node in da2node.items():
            if node not in G.nodes():
                continue
            idx += 1
            node2idx[node] = idx
            idx2node[idx] = node
        # label the rest
        for node in G.nodes():
            if node in node2idx:
                continue
            idx += 1
            node2idx[node] = idx
            idx2node[idx] = node
        dump_file(path_node2idx, node2idx)
        dump_file(path_idx2node, idx2node)
        return node2idx, idx2node

    @staticmethod
    def _graph_compression(G, node2idx, name):
        path = './data/trt_instance/{}_comp.pkl'.format(name)
        G_new, load_succeed = load_file(path)
        if load_succeed:
            return G_new
        edges = []
        for fnode, tnode in G.edges():
            edge = (node2idx[fnode], node2idx[tnode], G[fnode][tnode])
            edges.append(edge)
        G_new = nx.DiGraph()
        G_new.add_edges_from(edges)
        dump_file(path, G_new)
        return G_new

    @staticmethod
    def _od_compression(od_pairs, node2idx, name):
        path = './data/trt_instance/od_pairs_{}_comp.pkl'.format(name)
        od_comp, load_succeed = load_file(path)
        if load_succeed:
            return od_comp
        od_comp = {node2idx[orig]: {node2idx[des]: od_pairs[orig][des] for des in od_pairs[orig]} for orig in od_pairs}
        dump_file(path, od_comp)
        return od_comp

    @staticmethod
    def _extract_od_pairs(od_all, od_conn):
        path = './data/trt_instance/od_pairs_remained.pkl'
        od_remained, load_succeed = load_file(path)
        if load_succeed:
            return od_remained
        od_remained = {orig: {des: length for des, length in destination.items()
                               if (orig not in od_conn) or ((orig in od_conn) and (des not in od_conn[orig]))}
                       for orig, destination in od_all.items()}
        od_remained = {orig: destination for orig, destination in od_remained.items() if len(destination) > 0}
        dump_file(path, od_remained)
        return od_remained

    @staticmethod
    def _build_current_nx(G):
        path = './data/trt_network/final/G_curr.pkl'
        G_curr, load_succeed = load_file(path)
        if load_succeed:
            return G_curr
        edges = []
        fake_intersect = set(list(pd.read_csv('./data/trt_sig_intersect/fake_intersections.csv')['FNODE'].values))
        for fnode, tnode in G.edges():
            if ((G[fnode][tnode]['lts'] > 2) or (G.nodes[fnode]['lts'] > 2)) and (fnode not in fake_intersect):
                cost = 100
            else:
                cost = G[fnode][tnode]['time']
            edges.append((fnode, tnode, {'time': cost}))
        G_curr = nx.DiGraph()
        G_curr.add_edges_from(edges)
        dump_file(path, G_curr)
        return G_curr

    @staticmethod
    def _add_node_feature(G):
        node2sig, _ = load_file('./data/trt_sig_intersect/node2sig.pkl')
        attrs = {}
        for node in G.nodes:
            lts = 2
            if not node2sig[node]:
                for _, tnode in G.out_edges(node):
                    if G[node][tnode]['lts'] >= 3:
                        lts = 3
                        break
                for fnode, _ in G.in_edges(node):
                    if G[fnode][node]['lts'] >= 3:
                        lts = 3
                        break
            attrs[node] = lts
        nx.set_node_attributes(G, attrs, 'lts')
        return G

    @staticmethod
    def _orig2des_length(orig, destinations, T, G):
        if orig not in G.nodes():
            return {}
        reachable = single_source_dijkstra_path_length(G=G, source=orig, cutoff=T, weight='time')
        return {des: val for des, val in reachable.items() if des in destinations and des != orig}

    @staticmethod
    def _load_network_nx():
        path = './data/trt_network/final/G_simplified.pkl'
        G, _ = load_file(path)
        return G

    @staticmethod
    def _load_da2node():
        path_da2node = './data/trt_da/da2node.pkl'
        da2node, _ = load_file(path_da2node)
        return da2node

    @staticmethod
    def _load_sig_costs(G):
        path = './data/trt_sig_intersect/node2sig.pkl'
        node2sig, _ = load_file(path)
        fake_intersect = set(list(pd.read_csv('./data/trt_sig_intersect/fake_intersections.csv')['FNODE'].values))
        return {node: 1 for node in G.nodes() if not node2sig[node] and node not in fake_intersect}

    @staticmethod
    def _load_pop(da2node):
        path_da = './data/trt_da/da.pkl'
        df_da, _ = load_file(path_da)
        dauid = df_da['DAUID'].values
        pop = df_da['pop'].values
        node_pop = {da2node[da]: 0 for da in dauid}
        for idx in range(len(pop)):
            node_pop[da2node[dauid[idx]]] += pop[idx]
        return node_pop

    @staticmethod
    def _load_job():
        path_da2node = './data/trt_da/da2node.pkl'
        da2node, _ = load_file(path_da2node)
        df_da = pd.read_excel('./data/trt_da/pop_job_census.xls')
        dauid = df_da['DAUID'].values
        job = df_da['Census_job'].values
        node_job = {da2node[str(da)]: 0 for da in dauid}
        for idx in range(len(job)):
            node_job[da2node[str(dauid[idx])]] += job[idx]
        return node_job

    @staticmethod
    def _load_fake_intersection():
        path = './data/trt_sig_intersect/fake_intersections.csv'
        df = pd.read_csv(path)
        return set(list(df['FNODE'].values))

    @staticmethod
    def _gen_travel_time(G):
        return {(fnode, tnode): G[fnode][tnode]['time'] for fnode, tnode in G.edges()}

    # def _dict2shp(self, proj_ends):
    #     print('Generating a shapefile for the projects')
    #     path_node2loc = './data/trt_network/intermediate/node_loc.pkl'
    #     node2loc, _ = load_file(path_node2loc)
    #     tnode, fnode, length, time, geometry = [], [], [], [], []
    #     for f, t in proj_ends:
    #         tnode.append(t)
    #         fnode.append(f)
    #         time.append(proj_ends[(f, t)])
    #         length.append(self._time2distance(proj_ends[(f, t)]))
    #         geometry.append(LineString([node2loc[f], node2loc[t]]))
    #     df = pd.DataFrame({'fnode': fnode, 'tnode': tnode, 'time': time, 'length': length})
    #     gdf = gpd.GeoDataFrame(df, geometry=geometry)
    #     gdf.to_file(driver='ESRI Shapefile', filename='./data/trt_instance/project_shapefile/proj.shp')

    @staticmethod
    def _time2distance(t):
        return t * 15000 / 60

    @staticmethod
    def _distance2time(d):
        return d * 60 / 15000

    @staticmethod
    def _filter_od(destination):
        emb_pairs, _ = load_file('./prob/trt/emb/emb_pairs32.pkl')
        feature, _ = load_file('./prob/trt/emb/feature32.pkl')
        args_old, _ = load_file('./prob/trt/arg_archive_wo_Yonge/args_wo_yonge.pkl')
        # remove od pairs w/o emb
        od_pairs_w_emb = [(orig, des) for orig in args_old['destinations'] for des in args_old['destinations'][orig]]
        od_pairs_w_emb = np.array(od_pairs_w_emb)[emb_pairs]
        destination_new = {}
        feature_idx = {}
        cnt = -1
        for orig, des in od_pairs_w_emb:
            cnt += 1
            if (orig not in destination) or (des not in destination[orig]):
                continue
            if orig in destination_new:
                destination_new[orig].update({des: 1})
                feature_idx[orig].update({des: cnt})
            else:
                destination_new[orig] = {des: 1}
                feature_idx[orig] = {des: cnt}
        feature_new = [feature[feature_idx[orig][des]] for orig in destination_new for des in destination_new[orig]]
        feature_new = np.array(feature_new)
        dump_file('./prob/trt/emb/final/emb.pkl', feature_new)
        return destination_new
