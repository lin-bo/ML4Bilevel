#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin


from utils.instance_generator import ClusterGridGenerator
from embedding.embedding import DeepWalk, DeepWalk_variant, DeepWalk_utility, ScenarioSampler, build_ancillary_graph, build_ancillary_graph_variant, ScenarioSamplerRoute, build_ancillary_graph_utility
from utils.functions import load_file
from utils.sample_management import gen_active_samples, gen_active_samples_variant, gen_active_weighted_samples


def prob_initialization(width, n_orig, dim=32):
    # set parameters
    P = 25
    U = 10
    n_scenarios = 5000
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
    # instance generation
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3,
                                     p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    # embedding training
    T, M = 60, 70
    SS = ScenarioSampler(P=P, U=U, save=True)
    conn_matrix, time_matrix = SS.sample_train(args=args, n=n_scenarios, ins_name=ins_name)
    _, _ = SS.sample_test(args=args, n=n_scenarios, ins_name=ins_name)
    weights = build_ancillary_graph(ins_name=ins_name, suffix=suffix, time_matrix=time_matrix,
                                    conn_matrix=conn_matrix, threshold=0.9, save=True)
    DW = DeepWalk(random_seed=12, save=True)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=weights, walk_per_node=50, walk_length=20, dim=dim)
    # # initialize samples
    seeds = [0, 12, 23, 34, 45, 56, 67, 78, 89, 90]
    sizes = [0.005, 0.01, 0.02, 0.03, 0.04, 0.05]
    for sample_method in ['uniform', 'pmedian', 'pcenter']:
        gen_active_samples(width=width, n_orig=n_orig, feature=feature, args=args,
                           sizes=sizes, seeds=seeds, pmedian_penalities=[0], sample_method=sample_method, dim=dim)
    # samples = gen_samples(width=width, n_orig=n_orig, feature=feature, args=args, sizes=sizes, seeds=seeds, dim=dim)
    gen_active_weighted_samples(width=width, n_orig=n_orig, feature=feature, args=args, sizes=sizes, seeds=seeds, pmedian_penalities=[0])


def prob_variant_initialization(width=6, n_orig=72, dim=16, variant_type='exp'):
    # set parameters
    P, U, n_scenarios = 25, 10, 5000
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
    time_matrix, _ = load_file('./prob/{}/emb/time_matrix-p25-u10-n5000.pkl'.format(ins_name))
    args, _ = load_file('./prob/{}/args_c.pkl'.format(ins_name))
    # we can re-use the time matrix generated previously
    weights = build_ancillary_graph_variant(ins_name=ins_name, suffix=suffix, time_matrix=time_matrix, save=True)
    DW = DeepWalk_variant(random_seed=12, save=True, variant_type=variant_type)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=weights, walk_per_node=50, walk_length=20, dim=dim)
    # two_dim_visual(multi2two_dim(feature), [], [])
    # generate samples
    seeds = [0, 12, 23, 34, 45, 56, 67, 78, 89, 90]
    sizes = [0.01, 0.02, 0.03, 0.04, 0.05]
    for sample_method in ['uniform', 'pmedian', 'pcenter']:
        gen_active_samples_variant(width=width, n_orig=n_orig, feature=feature, args=args, sizes=sizes, variant_type=variant_type,
                                   seeds=seeds, pmedian_penalities=[0], sample_method=sample_method, dim=dim)


def prob_utility_initialization(width=6, n_orig=72, dim=16):
    # set parameters
    alpha = 1.02
    P, n_scenarios = 30, 10000
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}_n{}'.format(P, n_scenarios)
    # initilize the instance
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70,
                                     p_sig=0.3, p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate_wroutes(save=True)
    # learn OD embedding
    SSR = ScenarioSamplerRoute(P=P, save=True)
    utility_matrix = SSR.sample_train(args=args, n=n_scenarios, ins_name=ins_name)
    weights = build_ancillary_graph_utility(ins_name=ins_name, suffix=suffix, utility_matrix=utility_matrix, n_neighbor=10, save=True)
    DW = DeepWalk_utility(random_seed=12, save=True)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=weights, walk_per_node=50, walk_length=20, dim=dim)
    # two_dim_visual(multi2two_dim(feature), [], [])
    # generate samples
    seeds = [0, 12, 23, 34, 45, 56, 67, 78, 89, 90]
    sizes = [0.01, 0.02, 0.03, 0.04, 0.05]
    for sample_method in ['uniform', 'pmedian', 'pcenter']:
        gen_active_samples_variant(width=width, n_orig=n_orig, feature=feature, args=args, sizes=sizes, variant_type='utility',
                                   seeds=seeds, pmedian_penalities=[0], sample_method=sample_method, dim=dim)


if __name__ == '__main__':
    # initialize problems
    prob_initialization(width=6, n_orig=72, dim=16)
    prob_variant_initialization(variant_type='exp')
    prob_variant_initialization(variant_type='linear')
    prob_variant_initialization(variant_type='rec')
    prob_utility_initialization(width=6, n_orig=72, dim=16)
