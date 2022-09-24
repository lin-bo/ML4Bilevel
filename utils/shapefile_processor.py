#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

import geopandas as gpd
import pandas as pd
import numpy as np
import networkx as nx
from networkx.algorithms.shortest_paths.weighted import single_source_dijkstra_path_length
from utils.functions import load_file, dump_file, des2od
from shapely.geometry import Point, LineString
from tqdm import tqdm


class RoadNetworkSimplifier:

    def __init__(self):
        pass

    def simplify(self, fast_retrive=True, gen_shp=False):
        path = './data/trt_network/final/G_simplified.pkl'
        if fast_retrive:
            G, load_succeed = load_file(path)
            if load_succeed:
                return G
        # load raw network data
        df = self._load_network_shapefile()
        print('# of edges in the shapefile: {}'.format(len(df)))
        node2loc = self._gen_node2loc(df)
        node2sig = self._gen_node2sig(node2loc)
        art_inter = self._gen_art_inter(node2loc)
        print('# of nodes in the shapefile: {}'.format(len(node2loc)))
        print('# of signalized nodes in the shapefile: {}'.format(np.sum(list(node2sig.values()))))
        print('# of arterial intersection nodes: {}'.format(len(art_inter)))
        # search critical nodes where a road diverges
        critical_nodes = self._get_critical_nodes(df=df)
        print('# of critical nodes: {}'.format(len(critical_nodes)))
        # project da centroids to the road network
        df_da = self._load_da()
        da2node = self._project_da(df_da, node2loc)
        print('# of DAs in Toronto: {}'.format(len(df_da)))
        # build nx graph based on the network
        G = self._df2nx(df)
        G = self._add_node_feature(G, node2sig)
        print('# of edges in the initial graph: {}'.format(G.number_of_edges()))
        print('# of nodes in the initial graph: {}'.format(G.number_of_nodes()))
        artnode = self._load_artnode()
        print('# of nodes on arterial: {}'.format(len(artnode)))
        da2art_edges = self._build_da2art_connection(da2node, G, artnode)
        print('# of da-to-arterial edges: {}'.format(len(da2art_edges)))
        art2da_edges = self._build_art2da_connection(da2node, G, artnode)
        # manually add art2art edges
        art2art2_edges = self._add_art2art_connection()
        print('# of arterial-to-da edges: {}'.format(len(art2da_edges)))
        stopping_nodes = self._gen_stopping_node(critical_nodes, da2node, da2art_edges, art2da_edges)
        print('# of stopping nodes: {}'.format(len(stopping_nodes)))
        G = self._gen_simplified_nx(G, stopping_nodes, artnode, da2art_edges, art2da_edges, art2art2_edges)
        print('# of edges in the simplified graph: {}'.format(G.number_of_edges()))
        print('# of nodes in the simplified graph: {}'.format(G.number_of_nodes()))
        print('# of DAs in the simplified graph: {}'.format(self._check_da_existance(da2node, G)))
        if gen_shp:
            self.nx2shp(G, node2loc)
        return G

    @staticmethod
    def nx2shp(G, node2loc):
        print('Generating simplified shape file')
        tnode, fnode, lts, length, geometry = [], [], [], [], []
        for f, t in G.edges():
            tnode.append(t)
            fnode.append(f)
            lts.append(G[f][t]['lts'])
            length.append(G[f][t]['time'])
            geometry.append(LineString([node2loc[f], node2loc[t]]))
        df = pd.DataFrame({'fnode': fnode, 'tnode': tnode, 'lts': lts, 'time': length})
        gdf = gpd.GeoDataFrame(df, geometry=geometry)
        gdf.to_file(driver='ESRI Shapefile', filename='./data/trt_network/simplified_shapefile/simplified.shp')

    @staticmethod
    def _load_network_shapefile():
        path = './data/trt_network/intermediate/trt_network.pkl'
        df, load_succeed = load_file(path)
        if load_succeed:
            return df
        shapefile_path = './data/trt_network/lts4.shp'
        # read shape file
        df = pd.DataFrame(gpd.read_file(shapefile_path))
        # filter ramps
        df = df[~df['LF_NAME'].str.contains('Ramp')]
        # select useful columns
        df = df[['GEO_ID', 'LFN_ID', 'FNODE', 'TNODE', 'ONE_WAY_DI', 'LTS', 'length_in_', 'geometry']]
        df.columns = ['GEO_ID', 'LFN_ID', 'FNODE', 'TNODE', 'ONE_WAY_DI', 'LTS', 'Length', 'geometry']
        # remove cycles
        df = df[df['FNODE'] != df['TNODE']]
        dump_file(path, df)
        return df

    @staticmethod
    def _load_da():
        path = './data/trt_da/da.pkl'
        df, load_succeed = load_file(path)
        if load_succeed:
            return df
        da_path = './data/trt_da/DA_centroid.shp'
        df = pd.DataFrame(gpd.read_file(da_path))
        df = df[['DAUID', 'pop_census', 'Expanded_1', 'geometry']]
        df.columns = ['DAUID', 'pop', 'job', 'geometry']
        dump_file(path, df)
        return df

    @staticmethod
    def _load_artnode():
        path = './data/trt_arterial/artnode.pkl'
        artnode, load_succeed = load_file(path)
        if load_succeed:
            return artnode
        artnode = set([])
        art_path = './data/trt_arterial/trt_arterial.shp'
        df_art = pd.DataFrame(gpd.read_file(art_path))
        artnode.update(df_art['TNODE'].values)
        artnode.update(df_art['FNODE'].values)
        dump_file(path, artnode)
        return artnode

    @staticmethod
    def _get_critical_nodes(df):
        '''
        find critical nodes in the network. The definition of critical nodes: nodes that are connected with multiple
        roads or nodes that have only outgoing or incoming edges (not both).
        :param df:
        :return:
        '''
        path = './data/trt_network/intermediate/critical_nodes.pkl'
        # check if we have already generated these nodes
        critical_nodes, load_succeed = load_file(path)
        if load_succeed:
            return critical_nodes
        # we have to consider both from and to nodes as some nodes appear only once
        df1 = df[['FNODE', 'LFN_ID', 'GEO_ID']]
        df2 = df[['TNODE', 'LFN_ID', 'GEO_ID']]
        df2.columns = ['FNODE', 'LFN_ID', 'GEO_ID']
        df3 = pd.concat([df1, df2], axis=0)
        df3.index = range(len(df3))
        # to avoid duplicated nodes because of the concatenation above
        nodes = df3.groupby(['FNODE', 'LFN_ID']).count()
        nodes['Cnt'] = 1
        nodes = nodes.reset_index(level=['FNODE', 'LFN_ID'])
        # find nodes connected with edges with multiple names
        nodes = nodes[['FNODE', 'Cnt']].groupby(['FNODE']).sum()
        nodes = nodes[nodes['Cnt'] > 1]
        critical_nodes = set(nodes.index)
        # to avoid duplicated nodes because of the concatenation above
        nodes = df3.groupby(['FNODE', 'GEO_ID']).count()
        nodes['Cnt'] = 1
        nodes = nodes.reset_index(level=['FNODE', 'GEO_ID'])
        # find nodes at which a road diverges (connected with at least 3 geo-id) as critical nodes
        nodes = nodes[['FNODE', 'Cnt']].groupby(['FNODE']).sum()
        nodes = nodes[nodes['Cnt'] > 2]
        critical_nodes.update(list(nodes.index))
        # add outgoing only nodes (usually the end of a road)
        tmp1 = df[['FNODE', 'GEO_ID']].groupby('FNODE').count()
        tmp1 = set(list(tmp1.index))
        tmp2 = df[['TNODE', 'GEO_ID']].groupby('TNODE').count()
        tmp2 = set(list(tmp2.index))
        # remove duplicate nodes
        critical_nodes.union(tmp1.difference(tmp2))
        # store in local drive
        dump_file(path, critical_nodes)
        return critical_nodes

    @staticmethod
    def _df2nx(df):
        path = './data/trt_network/intermediate/G_initial.pkl'
        G, load_succeed = load_file(path)
        if load_succeed:
            return G
        # retrieve data
        ends = df[['FNODE', 'TNODE']].values
        lts = df['LTS'].values
        length = df['Length'].values
        oneway = df['ONE_WAY_DI'].values
        # generate edges
        edges = []
        for idx in range(len(df)):
            attr = {}
            if ends[idx][0] == ends[idx][1]:
                continue
            if oneway[idx] in [0, 1]:
                attr['lts'] = lts[idx]
                attr['length'] = length[idx]
                edges.append((ends[idx][0], ends[idx][1], attr))
            if oneway[idx] in [0, -1]:
                attr['lts'] = lts[idx]
                attr['length'] = length[idx]
                edges.append((ends[idx][1], ends[idx][0], attr))
            if oneway[idx] not in [0, 1, -1]:
                print('one way error, row{}, oneway: {}'.format(idx, oneway[idx]))
        G = nx.DiGraph()
        G.add_edges_from(edges)
        dump_file(path, G)
        return G

    @staticmethod
    def _add_node_feature(G, node2sig):
        path = './data/trt_network/intermediate/G_node_lts.pkl'
        G_node_lts, load_succeed = load_file(path)
        if load_succeed:
            return G_node_lts
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
        dump_file(path, G)
        return G

    @staticmethod
    def _gen_node2loc(df):
        path = './data/trt_network/intermediate/node_loc.pkl'
        loc, load_succeed = load_file(path)
        if load_succeed:
            return loc
        loc = {}
        fnodes = df['FNODE'].values
        tnodes = df['TNODE'].values
        coords = df['geometry'].values
        for i in range(len(df)):
            fcoords, tcoords = Point(list(coords[i].coords)[0]), Point(list(coords[i].coords)[-1])
            if fnodes[i] not in loc:
                loc[fnodes[i]] = fcoords
            if tnodes[i] not in loc:
                loc[tnodes[i]] = tcoords
        dump_file(path, loc)
        return loc

    @staticmethod
    def _gen_node2sig(node_loc):
        path = './data/trt_sig_intersect/node2sig.pkl'
        node2sig, load_succeed = load_file(path)
        if load_succeed:
            return node2sig
        sig_poly_path = './data/trt_sig_intersect/sig_inter_polygon.shp'
        # read shape file
        df = pd.DataFrame(gpd.read_file(sig_poly_path))
        polys = df['geometry'].values
        node2sig = {idx: False for idx, _ in node_loc.items()}
        for node, loc in tqdm(node_loc.items()):
            for p in polys:
                if p.contains(loc):
                    node2sig[node] = True
                    break
        dump_file(path, node2sig)
        return node2sig

    @staticmethod
    def _gen_art_inter(node_loc):
        path = './data/trt_arterial/art_inter.pkl'
        art_inter, load_succeed = load_file(path)
        if load_succeed:
            return art_inter
        art_inter_poly_path = './data/trt_arterial/arterial_conjunctions.shp'
        # read shape file
        df = pd.DataFrame(gpd.read_file(art_inter_poly_path))
        polys = df['geometry'].values
        art_inter = set([])
        for node, loc in tqdm(node_loc.items()):
            for p in polys:
                if p.contains(loc):
                    art_inter.add(node)
                    break
        dump_file(path, art_inter)
        return art_inter

    def _project_da(self, df_da, node_loc):
        path = './data/trt_da/da2node.pkl'
        da2node, load_succeed = load_file(path)
        if load_succeed:
            return da2node
        da2node = {}
        das = df_da['DAUID'].values
        coords = df_da['geometry'].values
        for i in tqdm(range(len(df_da))):
            da2node[das[i]] = self._search_nn(coords[i], node_loc)
        dump_file(path, da2node)
        return da2node

    @staticmethod
    def _search_nn(loc, node_loc):
        min_dist = 1e10
        nn = -1
        for node, coord in node_loc.items():
            dist = loc.distance(coord)
            if dist < min_dist:
                min_dist = dist
                nn = node
        if nn == -1:
            raise ValueError('nearest node not found')
        else:
            return nn

    def _build_da2art_connection(self, da2node, G, artnode):
        path = './data/trt_network/intermediate/da2art_edges.pkl'
        edges, load_succeed = load_file(path)
        if load_succeed:
            return edges
        G_da2art = self._build_da2node_graph(G, artnode)
        edges = []
        for da, node in da2node.items():
            edges += self._travel2art(node, G_da2art, artnode)
        dump_file(path, edges)
        return edges

    @staticmethod
    def _build_da2node_graph(G, art_node):
        edges = []
        for fnode, tnode in G.edges():
            if (fnode in art_node) or (G[fnode][tnode]['lts'] > 2) or (G.nodes[fnode]['lts'] >= 3):
                cost = 1e8
            else:
                cost = G[fnode][tnode]['length']
            edges.append((fnode, tnode, {'cost': cost}))
        G_da2art = nx.DiGraph()
        G_da2art.add_edges_from(edges)
        return G_da2art

    def _travel2art(self, node, G, artnode):
        lengths = single_source_dijkstra_path_length(G=G, source=node, cutoff=1e5, weight='cost')
        reachable_art = [(node, idx, {'time': self._distance2time(val), 'lts': 1}) for idx, val in lengths.items()
                         if idx in artnode and idx != node]
        return reachable_art

    def _build_art2da_connection(self, da2node, G, artnode):
        path = './data/trt_network/intermediate/art2da_edges.pkl'
        edges, load_succeed = load_file(path)
        if load_succeed:
            return edges
        G_art2da = self._build_art2da_graph(G, artnode)
        edges = []
        danode = set(list(da2node.values()))
        for node in tqdm(artnode):
            edges += self._travel2da(node, G_art2da, danode)
        dump_file(path, edges)
        return edges

    @staticmethod
    def _build_art2da_graph(G, art_node):
        edges = []
        for fnode, tnode in G.edges():
            if (tnode in art_node) or (G[fnode][tnode]['lts'] > 2) or (fnode not in art_node and G.nodes[fnode]['lts'] >= 3):
                cost = 1e8
            else:
                cost = G[fnode][tnode]['length']
            edges.append((fnode, tnode, {'cost': cost}))
        G_art2da = nx.DiGraph()
        G_art2da.add_edges_from(edges)
        return G_art2da

    def _travel2da(self, node, G, danode):
        if node not in G.nodes:
            return []
        lengths = single_source_dijkstra_path_length(G=G, source=node, cutoff=1e5, weight='cost')
        reachable_art = [(node, idx, {'time': self._distance2time(val), 'lts': 1}) for idx, val in lengths.items()
                         if idx in danode and idx != node]
        return reachable_art

    @staticmethod
    def _add_art2art_connection():
        edges = [(13463374, 30057484, {'time': 1.988762, 'lts': 1}),
                 (30057484, 13463374, {'time': 1.988762, 'lts': 1}),
                 (30003677, 13463374, {'time': 2.680890, 'lts': 1}),
                 (13463374, 30003677, {'time': 2.680890, 'lts': 1}),
                 (13461841, 30003689, {'time': 0.889600, 'lts': 1}),
                 (30003689, 13461841, {'time': 0.889600, 'lts': 1}),
                 (30003689, 30099089, {'time': 1.058440, 'lts': 1}),
                 (30099089, 30003689, {'time': 1.058440, 'lts': 1}),
                 (30099089, 30099062, {'time': 0.039780, 'lts': 1}),
                 (30099062, 30099089, {'time': 0.039780, 'lts': 1}),
                 (30099089, 30099036, {'time': 2.911300, 'lts': 1}),
                 (30099036, 30099089, {'time': 2.911300, 'lts': 1}),
                 (13462945, 13463078, {'time': 0.363100, 'lts': 1}),
                 (13463078, 13462945, {'time': 0.363100, 'lts': 1}),
                 (13462959, 13463068, {'time': 0.309700, 'lts': 1}),
                 (13463068, 13462959, {'time': 0.309700, 'lts': 1}),
                 (30071298, 30015319, {'time': 3.173900, 'lts': 1}),
                 (30015319, 30071298, {'time': 3.173900, 'lts': 1}),
                 (13463374, 30071298, {'time': 3.485920, 'lts': 1}),
                 (30071298, 13463374, {'time': 3.485920, 'lts': 1})
                 ]
        return edges

    @staticmethod
    def _gen_stopping_node(critical_nodes, da2node, da2art_edges, art2da_edges):
        path = './data/trt_network/intermediate/stopping_node.pkl'
        stopping_nodes, load_succeed = load_file(path)
        if load_succeed:
            return stopping_nodes
        stopping_nodes = critical_nodes.copy()
        stopping_nodes.update(list(da2node.values()))
        stopping_nodes.update([art for _, art, _ in da2art_edges])
        stopping_nodes.update([art for art, _, _ in art2da_edges])
        dump_file(path, stopping_nodes)
        return stopping_nodes

    def _gen_simplified_nx(self, G, stopping_node, artnode, da2art_edges, art2da_edges, art2art2_edges):
        path = './data/trt_network/final/G_simplified.pkl'
        G_new, load_succeed = load_file(path)
        if load_succeed:
            return G_new
        edges = da2art_edges + art2da_edges
        for node in stopping_node:
            if node not in G.nodes():
                continue
            edges += self._find_combined_edges(node, G, stopping_node, artnode)
        edges += art2art2_edges
        G_new = nx.DiGraph()
        G_new.add_edges_from(edges)
        dump_file(path, G_new)
        return G_new

    def _find_combined_edges(self, node, G, stopping_node, artnode):
        new_edges = []
        for fnode, tnode in G.out_edges(node):
            new_edge = self._search_forward(fnode, tnode, G, stopping_node)
            if (new_edge[1] in artnode) and (new_edge[0] in artnode) and (new_edge[0] != new_edge[1]):
                new_edges.append(new_edge)
        return new_edges

    def _search_forward(self, fnode, tnode, G, stopping_node):
        start_node = fnode
        length, lts = G[fnode][tnode]['length'], G[fnode][tnode]['lts']
        while tnode not in stopping_node:
            next_nodes = self._possible_next_nodes(G, fnode, tnode)
            if len(next_nodes) > 1:
                raise ValueError('direction {} -> {} should not have multiple possible next nodes'.format(fnode, tnode), next_nodes)
            elif len(next_nodes) == 0:
                break
            fnode, tnode = tnode, next_nodes[0]
            length += G[fnode][tnode]['length']
            lts = np.max([lts, G[fnode][tnode]['lts']])
        new_edge = (start_node, tnode, {'lts': lts, 'time': self._distance2time(length)})
        return new_edge

    @staticmethod
    def _possible_next_nodes(G, fnode, tnode):
        return [e2 for e1, e2 in G.out_edges(tnode) if e2 != fnode]

    @staticmethod
    def _distance2time(d):
        return d * 60 / 15000

    @staticmethod
    def _check_da_existance(da2node, G):
        cnt = 0
        for da, node in da2node.items():
            if node in G.nodes():
                cnt += 1
        return cnt


class ProjectLocator():

    def __init__(self):
        pass

    def find(self):
        G = self._build_art_graph()
        proj_ends = self._load_project_ends()
        node2idx = self._build_node2idx_dict()
        self._locate(G, proj_ends, node2idx)

    @staticmethod
    def _build_art_graph():
        # load data
        df_art = gpd.read_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/trt_arterial/trt_arterial.shp')
        ends = df_art[['FNODE', 'TNODE']].values
        length = df_art['length_in_'].values
        # create edges
        edges = []
        for idx, nodes in enumerate(ends):
            fnode, tnode = nodes
            edges.append((fnode, tnode, {'cost': length[idx]}))
            edges.append((tnode, fnode, {'cost': length[idx]}))
        # create the graph
        G = nx.DiGraph()
        G.add_edges_from(edges)
        return G

    @staticmethod
    def _load_project_ends():
        proj_ends, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/trt_instance/projends.pkl')
        return proj_ends

    @staticmethod
    def _build_node2idx_dict():
        df_art = gpd.read_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/trt_arterial/trt_arterial.shp')
        ends = df_art[['FNODE', 'TNODE']].values
        node2idx = {}
        for idx in range(len(ends)):
            node2idx[(ends[idx][0], ends[idx][1])] = idx
        return node2idx

    def _locate(self, G, proj_ends, node2idx):
        projects = []
        for orig, des in proj_ends:
            path = nx.dijkstra_path(G=G, source=orig, target=des, weight='cost')
            edges = self._find_edges(path, node2idx)
            projects.append(edges)
        dump_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/trt_instance/proj2artid.pkl', projects)

    @staticmethod
    def _find_edges(path, node2idx):
        edges = []
        for idx, node in enumerate(path[:-1]):
            if (node, path[idx+1]) in node2idx:
                edges.append(node2idx[(node, path[idx+1])])
            elif (path[idx+1], node) in node2idx:
                edges.append(node2idx[(path[idx+1], node)])
            else:
                raise ValueError('edge not found')
        return edges


class EquityAccessInterplay():

    def __init__(self):
        pass

    def check(self):
        self.check_curr_acc()

    @staticmethod
    def check_curr_acc():
        # load dataset
        conn_pairs, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/trt_instance/od_pairs_conn.pkl')
        remained_pairs, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/trt_instance/od_pairs_remained.pkl')
        args, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/prob/trt/args_adj.pkl')
        node2score, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/node2score.pkl')
        pop = args['populations']
        curr_acc = {}
        # calculate current accessibility
        for orig, destinations in conn_pairs.items():
            curr_acc[orig] = 0
            for des in destinations:
                curr_acc[orig] += pop[des]
        # calculate accessibility at LTS4
        tot_acc = curr_acc.copy()
        for orig, destinations in remained_pairs.items():
            if orig not in tot_acc:
                tot_acc[orig] = 0
            for des in destinations:
                tot_acc[orig] += pop[des]
        # connect with ON marginalized index
        data = []
        for orig, acc in tot_acc.items():
            tmp = curr_acc[orig] if orig in curr_acc else 0
            data.append([orig, tmp, acc, node2score[orig]])
        df = pd.DataFrame(data, columns=['DA', 'curr_acc', 'tot_acc', 'score'])
        df['Marginalization Group'] = pd.qcut(df['score'], 5, labels=False) + 1
        # partition by marginalization groups
        marg_groups = {}
        for i in range(1, 6):
            marg_groups[i] = df[df['Marginalization Group'] == i]['DA'].values.tolist()
        group = df['Marginalization Group'].values
        nodes = df['DA'].values
        node2group = {node: group[idx] for idx, node in enumerate(nodes)}
        for orig in tot_acc:
            if orig not in curr_acc:
                curr_acc[orig] = 0
        # save
        df.to_csv('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/acc_equity.csv', index=False)
        dump_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/curr_acc.pkl', curr_acc)
        dump_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/tot_acc.pkl', tot_acc)
        dump_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/marg_groups.pkl', marg_groups)
        dump_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/node2group.pkl', node2group)
        # load dataset
        conn_pairs, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/trt_instance/od_pairs_conn.pkl')
        remained_pairs, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/trt_instance/od_pairs_remained.pkl')
        args, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/prob/trt/args_adj.pkl')
        node2score, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/node2score.pkl')
        pop = args['populations']
        curr_acc = {}
        # calculate current accessibility
        for orig, destinations in conn_pairs.items():
            curr_acc[orig] = 0
            for des in destinations:
                curr_acc[orig] += pop[des]
        # calculate accessibility at LTS4
        tot_acc = curr_acc.copy()
        for orig, destinations in remained_pairs.items():
            if orig not in tot_acc:
                tot_acc[orig] = 0
            for des in destinations:
                tot_acc[orig] += pop[des]
        # connect with ON marginalized index
        data = []
        for orig, acc in tot_acc.items():
            tmp = curr_acc[orig] if orig in curr_acc else 0
            data.append([orig, tmp, acc, node2score[orig]])
        df = pd.DataFrame(data, columns=['DA', 'curr_acc', 'tot_acc', 'score'])
        df['Marginalization Group'] = pd.qcut(df['score'], 5, labels=False) + 1
        # partition by marginalization groups
        marg_groups = {}
        for i in range(1, 6):
            marg_groups[i] = df[df['Marginalization Group'] == i]['DA'].values.tolist()
        group = df['Marginalization Group'].values
        nodes = df['DA'].values
        node2group = {node: group[idx] for idx, node in enumerate(nodes)}
        for orig in tot_acc:
            if orig not in curr_acc:
                curr_acc[orig] = 0
        # save
        df.to_csv('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/acc_equity.csv', index=False)
        dump_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/curr_acc.pkl', curr_acc)
        dump_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/tot_acc.pkl', tot_acc)
        dump_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/marg_groups.pkl', marg_groups)
        dump_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/on_marg_index/node2group.pkl', node2group)


def add_equity_data(potential):
    # load dataset
    conn_pairs, _ = load_file('./data/trt_instance/od_pairs_conn.pkl')
    args, _ = load_file('./prob/trt/args.pkl')
    od_pairs = des2od(args['destinations'])
    node2score, _ = load_file('./data/on_marg_index/node2score.pkl')
    pop = args[potential]
    curr_acc = {}
    # calculate current accessibility
    for orig, destinations in conn_pairs.items():
        curr_acc[orig] = 0
        for des in destinations:
            curr_acc[orig] += pop[des]
    # calculate accessibility at LTS4
    tot_acc = curr_acc.copy()
    for orig, destinations in args['destinations'].items():
        if orig not in tot_acc:
            tot_acc[orig] = 0
        for des in destinations:
            tot_acc[orig] += pop[des]
    # connect with ON marginalized index
    data = []
    for orig, t_acc in tot_acc.items():
        c_acc = curr_acc[orig] if orig in curr_acc else 0
        data.append([orig, c_acc, t_acc, node2score[orig]])
    df = pd.DataFrame(data, columns=['DA', 'curr_acc', 'tot_acc', 'score'])
    df['Marginalization Group'] = pd.qcut(df['score'], 5, labels=False) + 1
    # partition by marginalization groups
    marg_groups = {}
    for i in range(1, 6):
        marg_groups[i] = df[df['Marginalization Group'] == i]['DA'].values.tolist()
    group = df['Marginalization Group'].values
    nodes = df['DA'].values
    node2group = {node: group[idx] for idx, node in enumerate(nodes)}
    for orig in tot_acc:
        if orig not in curr_acc:
            curr_acc[orig] = 0
    # save
    df.to_csv('./data/on_marg_index/acc_equity.csv', index=False)
    dump_file('./data/on_marg_index/curr_acc.pkl', curr_acc)
    dump_file('./data/on_marg_index/tot_acc.pkl', tot_acc)
    dump_file('./data/on_marg_index/marg_groups.pkl', marg_groups)
    dump_file('./data/on_marg_index/node2group.pkl', node2group)


if __name__ == '__main__':
    # PL = ProjectLocator()
    # PL.find()
    EQI = EquityAccessInterplay()
    EQI.check()
