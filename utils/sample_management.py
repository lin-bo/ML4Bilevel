#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

from embedding.coreset import greedy_kcenter, PMedianHeuristicSolver
from utils.od_samplers import naive_sampler
from utils.functions import dump_file, load_file, str2list, des2od
import pandas as pd
import multiprocessing
# import geopandas as gpd
# from shapely.geometry import LineString


def gen_samples(width, n_orig, feature, args, sizes, seeds):
    # initialization
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    sample_path = './prob/{}/samples/samples.pkl'.format(ins_name)
    samples, load_succeed = load_file(sample_path)
    pmedian_objs, _ = load_file('./prob/{}/samples/pmedian_obj.pkl'.format(ins_name))
    if not load_succeed:
        samples = {'uniform': {}, 'pcenter': {}, 'pmedian': {}}
    # generate
    pmedian_objs = {}
    for p in sizes:
        size = int(p * len(feature))
        pmedian, pcenter, uniform = {}, {}, {}
        objs = {}
        for seed in seeds:
            if p in samples['uniform'] and seed in samples['uniform'][p]:
                continue
            print('\n******************')
            print('size: {} ({} %), seed: {}'.format(size, p * 100, seed))
            # solve on the p-median sample
            pmedian_solver = PMedianHeuristicSolver(n_init=10, n_swap=100, n_term=30, quiet=True, feature=feature, distance_metric='cosine')
            pairs_pmedian, obj = pmedian_solver.solve(p=size, kappa=1)
            pmedian[seed] = pairs_pmedian
            print('  * p-median', obj, pairs_pmedian)
            objs[seed] = obj
            # p-center selection
            pairs_pcenter, _ = greedy_kcenter(feature=feature, n=size, k=1, repeat=200, tol=0.01, random_seed=seed)
            pcenter[seed] = pairs_pcenter
            print('  * p-center', pairs_pcenter)
            # uniform sampling
            pairs_uniform = naive_sampler(args=args, n=size, random_seed=seed).tolist()
            uniform[seed] = pairs_uniform
            print('  * uniform', pairs_uniform)
            # save
            samples['uniform'][p] = uniform.copy()
            samples['pcenter'][p] = pcenter.copy()
            samples['pmedian'][p] = pmedian.copy()
            pmedian_objs[p] = objs.copy()
            dump_file(path='./prob/{}/samples/samples.pkl'.format(ins_name), file=samples)
            dump_file(path='./prob/{}/samples/pmedian_obj.pkl'.format(ins_name), file=pmedian_objs)
    return samples


def gen_pmedian_samples(size, seed, feature, penalty, weight=[]):
    pmedian_solver = PMedianHeuristicSolver(n_init=10, n_swap=100, n_term=30, quiet=True, feature=feature,
                                            uneven_penalty=penalty, distance_metric='cosine', random_seed=seed, weight=weight)
    pairs, obj = pmedian_solver.solve(p=size, kappa=1)
    return pairs, obj


def gen_active_samples(width, n_orig, feature, args, sizes, seeds, pmedian_penalities, sample_method='uniform', dim=32):
    # initialization
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    sample_path = './prob/{}/samples/active{}_samples_{}.pkl'.format(ins_name, dim, sample_method)
    samples, load_succeed = load_file(sample_path)
    if not load_succeed:
        samples = {}
    # generate
    for p in sizes:
        size = int(p * len(feature))
        records = {}
        for seed in seeds:
            if p in samples and seed in samples[p]:
                continue
            print('\n******************')
            print('size: {} ({} %), seed: {}'.format(size, p * 100, seed))
            # solve on the p-median sample
            if sample_method == 'pmedian':
                params = [(size, seed, feature, penalty) for penalty in pmedian_penalities]
                pool = multiprocessing.Pool(4)
                res = pool.starmap(gen_pmedian_samples, params)
                pool.close()
                records[seed] = {penalty: res[idx][0] for idx, penalty in enumerate(pmedian_penalities)}
            elif sample_method == 'pcenter':
                # p-center selection
                pairs_pcenter, _ = greedy_kcenter(feature=feature, n=size, k=1, repeat=200, tol=0.01, random_seed=seed)
                records[seed] = pairs_pcenter
                print('  * p-center', pairs_pcenter)
            else:
                # uniform sampling
                pairs_uniform = naive_sampler(args=args, n=size, random_seed=seed).tolist()
                records[seed] = pairs_uniform
                print('  * uniform', pairs_uniform)
            # save
            samples[p] = records.copy()
            dump_file(path=sample_path, file=samples)
    return samples


def gen_active_samples_variant(width, n_orig, feature, args, sizes, seeds, pmedian_penalities,
                               sample_method='uniform', dim=32, variant_type='exp'):
    # initialization
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    sample_path = './prob/{}/samples/{}/active{}_samples_{}.pkl'.format(ins_name, variant_type, dim, sample_method)
    samples, load_succeed = load_file(sample_path)
    if not load_succeed:
        samples = {}
    # generate
    for p in sizes:
        size = int(p * len(feature))
        records = {}
        for seed in seeds:
            if p in samples and seed in samples[p]:
                continue
            print('\n******************')
            print('size: {} ({} %), seed: {}'.format(size, p * 100, seed))
            # solve on the p-median sample
            if sample_method == 'pmedian':
                params = [(size, seed, feature, penalty) for penalty in pmedian_penalities]
                pool = multiprocessing.Pool(4)
                res = pool.starmap(gen_pmedian_samples, params)
                pool.close()
                records[seed] = {penalty: res[idx][0] for idx, penalty in enumerate(pmedian_penalities)}
            elif sample_method == 'pcenter':
                # p-center selection
                pairs_pcenter, _ = greedy_kcenter(feature=feature, n=size, k=1, repeat=200, tol=0.0, random_seed=seed)
                records[seed] = pairs_pcenter
                print('  * p-center', pairs_pcenter)
            else:
                # uniform sampling
                pairs_uniform = naive_sampler(args=args, n=size, random_seed=seed).tolist()
                records[seed] = pairs_uniform
                print('  * uniform', pairs_uniform)
            # save
            samples[p] = records.copy()
            dump_file(path=sample_path, file=samples)
    return samples


def gen_active_weighted_samples(width, n_orig, feature, args, sizes, seeds, pmedian_penalities):
    # initialization
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    sample_path = './prob/{}/samples/active_samples_pmedian_weighted.pkl'.format(ins_name)
    samples, load_succeed = load_file(sample_path)
    if not load_succeed:
        samples = {}
    # generate
    weight = [args['populations'][des] for orig, des in des2od(args['destinations'])]
    for p in sizes:
        size = int(p * len(feature))
        records = {}
        for seed in seeds:
            if p in samples and seed in samples[p]:
                continue
            print('\n******************')
            print('size: {} ({} %), seed: {}'.format(size, p * 100, seed))
            # solve on the p-median sample
            params = [(size, seed, feature, penalty, weight) for penalty in pmedian_penalities]
            pool = multiprocessing.Pool(1)
            res = pool.starmap(gen_pmedian_samples, params)
            pool.close()
            records[seed] = {penalty: res[idx][0] for idx, penalty in enumerate(pmedian_penalities)}
            # save
            samples[p] = records.copy()
            dump_file(path=sample_path, file=samples)
    return samples


def visual_samples(samples):
    # load data
    node2loc, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/data/trt_network/intermediate/node_loc.pkl')
    # generate shapefile
    fnodes, tnodes, geometry = [], [], []
    for orig, des in samples:
        fnodes.append(orig)
        tnodes.append(des)
        geometry.append(LineString([node2loc[orig], node2loc[des]]))
    df = pd.DataFrame({'fnode': fnodes, 'tnode': tnodes})
    gdf = gpd.GeoDataFrame(df, geometry=geometry)
    gdf.to_file(driver='ESRI Shapefile',
                filename='/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/prob/trt/visualization/selected_pairs.shp')


if __name__ == '__main__':
    df = pd.read_csv('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/prob/trt/pmedian/res_4.csv')
    samples = str2list(df['medians'].values[-1])
    args, _ = load_file('/Users/bolin/Library/CloudStorage/OneDrive-UniversityofToronto/MaxANDP/prob/trt/args_adj.pkl')
    od_pairs = des2od(args['destinations'])
    samples = [od_pairs[int(idx)] for idx in samples]
    visual_samples(samples)
