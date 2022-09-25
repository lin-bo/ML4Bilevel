from utils.instance_generator import ClusterGridGenerator
from embedding.embedding import DeepWalk, DeepWalk_variant, DeepWalk_utility, add_position_feature, ScenarioSampler, build_ancillary_graph, build_ancillary_graph_variant, ScenarioSamplerRoute, build_ancillary_graph_utility
from embedding.coreset import greedy_kcenter, gen_argument, gen_argument_prod, find_neighbors, PMedianSolver, gen_feature_dict
from embedding.visualization import two_dim_visual, multi2two_dim
from utils.od_samplers import naive_sampler
from solver.continuous.benders import BendersSolverOptimalityCut, BendersRegressionSolver, BendersBaggingSolver, BendersSolverOptimalityCutVariant, BendersRegressionSolverVariant
from solver.route_choice.bilevel import BilevelSolver
from utils.functions import cal_con_obj, cal_utility_obj, des2od, dump_file, load_file, cal_od_stds
import pandas as pd
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


def instance_visual(width, n_orig):
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
    Generator.visualize(args)


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


def embedding_initialization_varaint_nsim_impact(width=6, n_orig=72, dim=16, variant_type='exp', n_scenarios=5000):
    # set parameters
    P, U = 25, 10
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    if variant_type == 'utility':
        args, _ = load_file('./prob/{}/args_r.pkl'.format(ins_name))
        suffix = 'p{}_n{}'.format(30, n_scenarios)
        SSR = ScenarioSamplerRoute(P=P, save=True)
        utility_matrix = SSR.sample_train(args=args, n=n_scenarios, ins_name=ins_name)
        utility_matrix = utility_matrix[:n_scenarios]
        weights = build_ancillary_graph_utility(ins_name=ins_name, suffix=suffix, utility_matrix=utility_matrix, n_neighbor=10, save=True)
        DW = DeepWalk_utility(random_seed=12, save=True)
        feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=weights, walk_per_node=50, walk_length=20, dim=dim)
    else:
        args, _ = load_file('./prob/{}/args_c.pkl'.format(ins_name))
        suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
        time_matrix, _ = load_file('./prob/{}/emb/time_matrix-p25-u10-n5000.pkl'.format(ins_name))
        time_matrix = time_matrix[:n_scenarios]
        # we can re-use the time matrix generated previously
        weights = build_ancillary_graph_variant(ins_name=ins_name, suffix=suffix, time_matrix=time_matrix, save=True)
        DW = DeepWalk_variant(random_seed=12, save=True, variant_type=variant_type)
        feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=weights, walk_per_node=50, walk_length=20, dim=dim)


def opt_sol(budget_scenarios):
    # set parameters
    width = 6
    n_orig = 36
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    # generate a continuous instance
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3,
                                     p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    records = []
    for budget_proj, budget_sig in budget_scenarios:
        print('\n==================')
        print('project budget: {}, signal budget: {}'.format(budget_proj, budget_sig))
        solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=True)
        new_projects, new_signals, t_full = solver.solve(args, budget_proj=budget_proj, budget_sig=budget_sig,
                                                         beta_1=0.001, regenerate=False, pareto=True, relax4cut=True,
                                                         weighted=False, quiet=False)
        obj_full, _ = cal_con_obj(args, new_projects, new_signals, solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'], M=args['travel_time_max']))
        print('full model -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_full, t_full))
        r = [budget_proj, budget_sig, obj_full, t_full]
        records.append(r)
        df = pd.DataFrame(records, columns=['budget_proj', 'budget_sig', 'obj_full', 't_full'])
        df.to_csv('./prob/{}/res/opt_sol.csv'.format(ins_name))


def budget_impact(budget_scenarios, n_repeat):
    # set parameters
    width = 6
    n_orig = 36
    P, U, n_scenarios = 30, 10, 5000
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
    seeds = [12, 23, 34, 45, 56, 67, 78, 89, 100]
    save = False
    # generate a continuous instance
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3,
                                     p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    # load features
    DW = DeepWalk(random_seed=12, save=save)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=[], walk_per_node=50, walk_length=20, dim=32)
    feature = add_position_feature(args, feature)
    records = []
    for budget_proj, budget_sig in budget_scenarios:
        print('\n==================')
        print('project budget: {}, signal budget: {}'.format(budget_proj, budget_sig))
        for seed in seeds[:n_repeat]:
            print('\n******************')
            print('seed: {}'.format(seed))
            # uniform sampling
            selected_pairs = naive_sampler(args=args, n=100, random_seed=seed)
            args_new = gen_argument(args, selected_pairs, [])
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_uniform_saa = solver.solve(args_new, budget_proj=budget_proj, budget_sig=budget_sig, beta_1=0.001,
                                                            regenerate=True, pareto=False, relax4cut=True, weighted=False, quiet=True)
            obj_uniform_saa, _ = cal_con_obj(args, new_projects, new_signals, solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'], M=args['travel_time_max']))
            print('uniform + SAA -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_uniform_saa, t_uniform_saa))
            # weighted uniform
            neighbors = find_neighbors(feature, selected_pairs, k=1)
            args_new = gen_argument(args, selected_pairs, neighbors)
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_uniform_knn = solver.solve(args_new, budget_proj=budget_proj,
                                                                 budget_sig=budget_sig, beta_1=0.001,
                                                                 regenerate=True, pareto=False, relax4cut=True,
                                                                 weighted=True, quiet=True)
            obj_uniform_knn, _ = cal_con_obj(args, new_projects, new_signals,
                                          solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                            M=args['travel_time_max']))
            print('uniform + KNN -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_uniform_knn, t_uniform_knn))
            # core set selection
            coreset, neighbors = greedy_kcenter(feature=feature, n=100, k=1, repeat=200, tol=0.01, random_seed=seed)
            args_new = gen_argument(args, coreset, neighbors)
            # core set method
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_pcenter_saa = solver.solve(args_new, budget_proj=budget_proj, budget_sig=budget_sig, beta_1=0.001,
                                                            regenerate=True, pareto=False, relax4cut=True, weighted=False, quiet=True)
            obj_pcenter_saa, _ = cal_con_obj(args, new_projects, new_signals,
                                     solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'], M=args['travel_time_max']))
            print('pcenter + SAA -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_pcenter_saa, t_pcenter_saa))
            # weighted core set method
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_pcenter_knn = solver.solve(args_new, budget_proj=budget_proj, budget_sig=budget_sig, beta_1=0.001,
                                                            regenerate=True, pareto=False, relax4cut=True, weighted=True, quiet=True)
            obj_pcenter_knn, _ = cal_con_obj(args, new_projects, new_signals,
                                     solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'], M=args['travel_time_max']))
            print('pcenter + KNN -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_pcenter_knn, t_pcenter_knn))
            # store the results
            r = [budget_proj, budget_sig, seed,
                 obj_uniform_saa, t_uniform_saa,
                 obj_uniform_knn, t_uniform_knn,
                 obj_pcenter_saa, t_pcenter_saa,
                 obj_pcenter_knn, t_pcenter_knn]
            records.append(r)
            df = pd.DataFrame(records, columns=['budget_proj', 'budget_sig', 'seed',
                                                'obj_uniform_saa', 't_uniform_saa',
                                                'obj_uniform_knn', 't_uniform_knn',
                                                'obj_pcenter_saa', 't_pcenter_saa',
                                                'obj_pcenter_knn', 't_pcenter_knn'])
            df.to_csv('./prob/{}/res/budget.csv'.format(ins_name))


def sample_size_impact(budget_proj, budget_sig, n_repeat, sizes):
    # set parameters
    width = 6
    n_orig = 36
    P, U, n_scenarios = 30, 10, 5000
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
    seeds = [12, 23, 34, 45, 56, 67, 78, 89, 100]
    save = False
    # generate a continuous instance
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3,
                                     p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    # load features
    DW = DeepWalk(random_seed=12, save=save)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=[], walk_per_node=50, walk_length=20, dim=32)
    feature = add_position_feature(args, feature)
    records = []
    for p in sizes:
        print('\n==================')
        size = int(p * len(feature))
        # initialize p-median solver
        print('initializing ...')
        pmsolver = PMedianSolver()
        _ = pmsolver.solve(points=feature, p=size)
        print('size: {} ({} %), fixed project budget: {}, fixed signal budget: {}'.format(size, p*100, budget_proj, budget_sig))
        for seed in seeds[:n_repeat]:
            print('\n******************')
            print('seed: {}'.format(seed))
            # uniform sampling
            selected_pairs = naive_sampler(args=args, n=size, random_seed=seed)
            args_new = gen_argument(args, selected_pairs, [])
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_uniform_saa = solver.solve(args_new, budget_proj=budget_proj,
                                                                budget_sig=budget_sig, beta_1=0.001,
                                                                regenerate=True, pareto=False, relax4cut=True,
                                                                weighted=False, quiet=True)
            obj_uniform_saa, _, _ = cal_con_obj(args, new_projects, new_signals,
                                         solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                           M=args['travel_time_max']))
            print('uniform + saa -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_uniform_saa, t_uniform_saa))
            # weighted uniform
            neighbors = find_neighbors(feature, selected_pairs, k=1)
            args_new = gen_argument(args, selected_pairs, neighbors)
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_uniform_knn = solver.solve(args_new, budget_proj=budget_proj,
                                                                 budget_sig=budget_sig, beta_1=0.001,
                                                                 regenerate=True, pareto=False, relax4cut=True,
                                                                 weighted=True, quiet=True)
            obj_uniform_knn, _, _ = cal_con_obj(args, new_projects, new_signals,
                                          solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                            M=args['travel_time_max']))
            print('uniform + knn -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_uniform_knn, t_uniform_knn))
            # pmedian
            pairs_kmedian = pmsolver.purturb(seed=seed)
            args_new = gen_argument(args, pairs_kmedian, neighbors)
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_pmedian_saa = solver.solve(args_new, budget_proj=budget_proj,
                                                                    budget_sig=budget_sig, beta_1=0.001,
                                                                    regenerate=True, pareto=False, relax4cut=True,
                                                                    weighted=False, quiet=True)
            obj_pmedian_saa, _, _ = cal_con_obj(args, new_projects, new_signals,
                                                solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                                  M=args['travel_time_max']))
            print('pmedian + saa -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_pmedian_saa, t_pmedian_saa))
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_pmedian_knn = solver.solve(args_new, budget_proj=budget_proj,
                                                                    budget_sig=budget_sig, beta_1=0.001,
                                                                    regenerate=True, pareto=False, relax4cut=True,
                                                                    weighted=True, quiet=True)
            obj_pmedian_knn, _, _ = cal_con_obj(args, new_projects, new_signals,
                                                solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                                  M=args['travel_time_max']))
            print('pmedian + knn -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_pmedian_knn, t_pmedian_knn))

            # p-center
            coreset, neighbors = greedy_kcenter(feature=feature, n=size, k=1, repeat=200, tol=0.01, random_seed=seed)
            args_new = gen_argument(args, coreset, neighbors)
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_pcenter_saa = solver.solve(args_new, budget_proj=budget_proj,
                                                                budget_sig=budget_sig, beta_1=0.001,
                                                                regenerate=True, pareto=False, relax4cut=True,
                                                                weighted=False, quiet=True)
            obj_pcenter_saa, _, _ = cal_con_obj(args, new_projects, new_signals,
                                         solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                           M=args['travel_time_max']))
            print('pcenter + saa -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_pcenter_saa, t_pcenter_saa))
            # weighted core set method
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_pcenter_knn = solver.solve(args_new, budget_proj=budget_proj,
                                                                 budget_sig=budget_sig, beta_1=0.001,
                                                                 regenerate=True, pareto=False, relax4cut=True,
                                                                 weighted=True, quiet=True)
            obj_pcenter_knn, _, _ = cal_con_obj(args, new_projects, new_signals,
                                             solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                               M=args['travel_time_max']))
            print('pcenter + knn -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_pcenter_knn, t_pcenter_knn))
            # store the results
            r = [budget_proj, budget_sig, seed, p, size,
                 obj_uniform_saa, t_uniform_saa,
                 obj_uniform_knn, t_uniform_knn,
                 obj_pmedian_saa, t_pmedian_saa,
                 obj_pmedian_knn, t_pmedian_knn,
                 obj_pcenter_saa, t_pcenter_saa,
                 obj_pcenter_knn, t_pcenter_knn]
            records.append(r)
            df = pd.DataFrame(records, columns=['budget_proj', 'budget_sig', 'seed', 'percent', 'size',
                                                'obj_uniform_saa', 't_uniform_saa',
                                                'obj_uniform_knn', 't_uniform_knn',
                                                'obj_pmedian_saa', 't_pmedian_saa',
                                                'obj_pmedian_knn', 't_pmedian_knn',
                                                'obj_pcenter_saa', 't_pcenter_saa',
                                                'obj_pcenter_knn', 't_pcenter_knn'])
            df.to_csv('./prob/{}/res/sample_size_budget{}_test.csv'.format(ins_name, budget_proj))


def embedding_performance_fixed_size(budget_scenarios, n_repeat):
    # set parameters
    width = 6
    n_orig = 36
    P, U, n_scenarios = 30, 10, 5000
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
    seeds = [12, 23, 34, 45, 56, 67, 78, 89, 100]
    save = False
    # generate a continuous instance
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3,
                                     p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    # load features
    # connection features + TSP faetures
    DW = DeepWalk(random_seed=12, save=save)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=[], walk_per_node=50, walk_length=20, dim=32)
    feature = add_position_feature(args, feature)
    # connection feature
    conn_feature = feature[:, :32].copy()
    # TSP feature
    tsp_feature = feature[:, 32:].copy()
    records = []
    for budget_proj, budget_sig in budget_scenarios:
        print('\n==================')
        print('size: {}, fixed project budget: {}, fixed signal budget: {}'.format(100, budget_proj, budget_sig))
        for seed in seeds[:n_repeat]:
            print('\n******************')
            print('seed: {}'.format(seed))
            selected_pairs = naive_sampler(args=args, n=100, random_seed=seed)
            # tsp
            neighbors = find_neighbors(tsp_feature, selected_pairs, k=1)
            args_new = gen_argument(args, selected_pairs, neighbors)
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_tsp = solver.solve(args_new, budget_proj=budget_proj,
                                                            budget_sig=budget_sig, beta_1=0.001,
                                                            regenerate=True, pareto=False, relax4cut=True,
                                                            weighted=True, quiet=True)
            obj_tsp, _ = cal_con_obj(args, new_projects, new_signals,
                                     solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                       M=args['travel_time_max']))
            print('tsp feature -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_tsp, t_tsp))
            # connection
            neighbors = find_neighbors(conn_feature, selected_pairs, k=1)
            args_new = gen_argument(args, selected_pairs, neighbors)
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_conn = solver.solve(args_new, budget_proj=budget_proj,
                                                            budget_sig=budget_sig, beta_1=0.001,
                                                            regenerate=True, pareto=False, relax4cut=True,
                                                            weighted=True, quiet=True)
            obj_conn, _ = cal_con_obj(args, new_projects, new_signals,
                                     solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                       M=args['travel_time_max']))
            print('connection feature -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_conn, t_conn))
            # combo
            neighbors = find_neighbors(feature, selected_pairs, k=1)
            args_new = gen_argument(args, selected_pairs, neighbors)
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_combo = solver.solve(args_new, budget_proj=budget_proj,
                                                              budget_sig=budget_sig, beta_1=0.001,
                                                              regenerate=True, pareto=False, relax4cut=True,
                                                              weighted=True, quiet=True)
            obj_combo, _ = cal_con_obj(args, new_projects, new_signals,
                                       solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                         M=args['travel_time_max']))
            print('combo uniform -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_combo, t_combo))
            # store the results
            r = [budget_proj, budget_sig, seed,
                 obj_tsp, t_tsp,
                 obj_conn, t_conn,
                 obj_combo, t_combo]
            records.append(r)
            df = pd.DataFrame(records, columns=['budget_proj', 'budget_sig', 'seed',
                                                'obj_tsp', 't_tsp',
                                                'obj_conn', 't_conn',
                                                'obj_combo', 't_combo'])
            df.to_csv('./prob/{}/res/emb_performance.csv'.format(ins_name))


def embedding_performance_fixed_budget(budget_proj, budget_sig, n_repeat, sizes):
    # set parameters
    width = 6
    n_orig = 36
    P, U, n_scenarios = 30, 10, 5000
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
    seeds = [12, 23, 34, 45, 56, 67, 78, 89, 100]
    save = False
    # generate a continuous instance
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3,
                                     p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    # load features
    # connection features + TSP faetures
    DW = DeepWalk(random_seed=12, save=save)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=[], walk_per_node=50, walk_length=20, dim=32)
    feature = add_position_feature(args, feature)
    # connection feature
    conn_feature = feature[:, :32].copy()
    # TSP feature
    tsp_feature = feature[:, 32:].copy()
    records = []
    for p in sizes:
        print('\n==================')
        size = int(p * len(feature))
        print('size: {} ({} %), fixed project budget: {}, fixed signal budget: {}'.format(size, p * 100, budget_proj, budget_sig))
        for seed in seeds[:n_repeat]:
            print('\n******************')
            print('seed: {}'.format(seed))
            selected_pairs = naive_sampler(args=args, n=size, random_seed=seed)
            # tsp
            neighbors = find_neighbors(tsp_feature, selected_pairs, k=1)
            args_new = gen_argument(args, selected_pairs, neighbors)
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_tsp = solver.solve(args_new, budget_proj=budget_proj,
                                                            budget_sig=budget_sig, beta_1=0.001,
                                                            regenerate=True, pareto=False, relax4cut=True,
                                                            weighted=True, quiet=True)
            obj_tsp, _ = cal_con_obj(args, new_projects, new_signals,
                                     solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                       M=args['travel_time_max']))
            print('tsp feature -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_tsp, t_tsp))
            # connection
            neighbors = find_neighbors(conn_feature, selected_pairs, k=1)
            args_new = gen_argument(args, selected_pairs, neighbors)
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_conn = solver.solve(args_new, budget_proj=budget_proj,
                                                             budget_sig=budget_sig, beta_1=0.001,
                                                             regenerate=True, pareto=False, relax4cut=True,
                                                             weighted=True, quiet=True)
            obj_conn, _ = cal_con_obj(args, new_projects, new_signals,
                                      solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                        M=args['travel_time_max']))
            print('connection feature -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_conn, t_conn))
            # combo
            neighbors = find_neighbors(feature, selected_pairs, k=1)
            args_new = gen_argument(args, selected_pairs, neighbors)
            solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
            new_projects, new_signals, t_combo = solver.solve(args_new, budget_proj=budget_proj,
                                                              budget_sig=budget_sig, beta_1=0.001,
                                                              regenerate=True, pareto=False, relax4cut=True,
                                                              weighted=True, quiet=True)
            obj_combo, _ = cal_con_obj(args, new_projects, new_signals,
                                       solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                         M=args['travel_time_max']))
            print('combo uniform -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_combo, t_combo))
            # store the results
            r = [budget_proj, budget_sig, seed, p, size,
                 obj_tsp, t_tsp,
                 obj_conn, t_conn,
                 obj_combo, t_combo]
            records.append(r)
            df = pd.DataFrame(records, columns=['budget_proj', 'budget_sig', 'seed', 'percent', 'size',
                                                'obj_tsp', 't_tsp',
                                                'obj_conn', 't_conn',
                                                'obj_combo', 't_combo'])
            df.to_csv('./prob/{}/res/emb_performance_fixedbudget.csv'.format(ins_name))


def solve_once(budget_proj, budget_sig, p):
    # set parameters
    width = 6
    n_orig = 36
    P, U, n_scenarios = 30, 10, 5000
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    # generate a continuous instance
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3,
                                     p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    size = int(p * len(des2od(args['destinations'])))
    # uniform sampling
    selected_pairs = naive_sampler(args=args, n=size, random_seed=0)
    args_new = gen_argument(args, selected_pairs, [])
    solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
    new_projects, new_signals, t_uniform = solver.solve(args_new, budget_proj=budget_proj, budget_sig=budget_sig, beta_1=0.001,
                                                        regenerate=False, pareto=True, relax4cut=True, weighted=False, quiet=False)
    obj_uniform, _ = cal_con_obj(args, new_projects, new_signals,
                                 solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'],
                                                   M=args['travel_time_max']))
    print('uniform -- obj: {:.4f} -- time: {:.4f} sec'.format(obj_uniform, t_uniform))


def complete_eval(width, n_orig, budgets, sizes, sample_method='uniform', solving_method='naive', dim=32):
    param_dict = {(6, 36): (30, 10, 5000), (6, 72): (25, 10, 5000)}
    # set parameters
    P, U, n_scenarios = param_dict[width, n_orig]
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
    seeds = [0, 12, 23, 34, 45, 56, 67, 78, 89, 90]
    save = False
    # load data
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3, p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    DW = DeepWalk(random_seed=12, save=save)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=[], walk_per_node=50, walk_length=20, dim=32)
    # initialize the samples
    if sample_method == 'pmedian_weighted':
        sample_path = './prob/{}/samples/active_samples_pmedian_weighted.pkl'.format(ins_name)
        tmp, _ = load_file(sample_path)
        samples = {}
        samples[sample_method] = tmp.copy()
    else:
        # samples = gen_samples(width=width, n_orig=n_orig, feature=feature, args=args, sizes=sizes, seeds=seeds)
        samples = gen_active_samples(width=width, n_orig=n_orig, feature=feature, args=args,
                           sizes=sizes, seeds=seeds, pmedian_penalities=[], sample_method=sample_method, dim=dim)
    # start the computation
    df_path = './prob/{}/res/complete/complete_{}_{}.pkl'.format(ins_name, sample_method, solving_method)
    df, load_succeed = load_file(df_path)
    if load_succeed:
        calculated = {(budget, size, seed) for budget, size, seed in df[['budget', 'size', 'seed']].values}
        records = df.values.tolist()
    else:
        calculated = {}
        records = []
    for budget_proj in budgets:
        for size in sizes:
            for seed in seeds:
                if (budget_proj, size, seed) in calculated:
                    continue
                if sample_method == 'pmedian':
                    selected_pairs = samples[size][seed][0]
                else:
                    selected_pairs = samples[size][seed]
                neighbors = find_neighbors(feature, selected_pairs, k=1)
                loss_bound = 0.3 * len(selected_pairs)
                if solving_method in ['naive', 'knn']:
                    args_new = gen_argument(args, selected_pairs, neighbors)
                    solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False)
                    weighted = True if solving_method == 'knn' else False
                    new_projects, new_signals, t, _ = solver.solve(args_new, budget_proj=budget_proj, budget_sig=0, beta_1=0.001, regenerate=True, pareto=True, relax4cut=True,
                                                                    weighted=weighted, quiet=True)
                elif solving_method == 'reg':
                    args_new = gen_argument(args, selected_pairs, neighbors, feature=feature)
                    solver = BendersRegressionSolver(ins_name=ins_name, save_model=False)
                    new_projects, new_signals, t, loss_bound = solver.solve(args_new, budget_proj=budget_proj, budget_sig=0, beta_1=0.001, regenerate=True, pareto=True, relax4cut=True,
                                                                weighted=False, quiet=True, insample_weight=1, reg_factor=32, loss_bound=loss_bound)
                else:
                    raise ValueError('solution method {} is not defined'.format(solving_method))
                obj, acc, _ = cal_con_obj(args, new_projects, new_signals, solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'], M=args['travel_time_max']))
                print('{} + {} (size: {}, seed: {}) -- obj: {:.4f} -- acc: {:.4f} -- time: {:.4f} sec'.format(sample_method, solving_method, size, seed, obj, acc, t))
                record = [budget_proj, size, seed, obj, acc, t, loss_bound, new_projects]
                records.append(record.copy())
                df = pd.DataFrame(records, columns=['budget', 'size', 'seed', 'obj', 'acc', 'time', 'loss_bound', 'projects'])
                dump_file(df_path, df)
                df.to_csv('./prob/{}/res/complete/complete_{}_{}.csv'.format(ins_name, sample_method, solving_method), index=False)


def complete_eval_variant(width, n_orig, budgets, sizes, variant_type='exp',
                          sample_method='uniform', solving_method='naive', dim=16):
    param_dict = {(6, 36): (30, 10, 5000), (6, 72): (25, 10, 5000)}
    # set parameters
    P, U, n_scenarios = param_dict[width, n_orig]
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
    seeds = [0, 12, 23, 34, 45, 56, 67, 78, 89, 90]
    # set manual params
    if variant_type == 'exp':
        manual_params = {'M': 60, 'T': 20, 'beta': [1, 0.75 / 20, 0.25 / 40]}
    elif variant_type == 'linear':
        manual_params = {'M': 60, 'T': 60, 'beta': [1, 1 / 60, 0]}
    elif variant_type == 'rec':
        manual_params = {'M': 60, 'T': 58, 'beta': [1, 0.001, 0.942 / 2]}
    else:
        raise ValueError('variant type {} not found'.format(variant_type))
    # load data
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3, p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    weights, _ = load_file('./prob/{}/emb/{}/ancillary_graph_weights_p25-u10-n5000.pkl'.format(ins_name, variant_type))
    DW = DeepWalk_variant(random_seed=12, save=True, variant_type=variant_type)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=weights, walk_per_node=50, walk_length=20, dim=dim)
    # initialize the samples
    samples, _ = load_file('./prob/{}/samples/{}/active16_samples_{}.pkl'.format(ins_name, variant_type, sample_method))
    # start the computation
    df_path = './prob/{}/res/complete_{}/complete_{}_{}.pkl'.format(ins_name, variant_type, sample_method, solving_method)
    df, load_succeed = load_file(df_path)
    if load_succeed:
        calculated = {(budget, size, seed) for budget, size, seed in df[['budget', 'size', 'seed']].values}
        records = df.values.tolist()
    else:
        calculated = {}
        records = []
    for budget_proj in budgets:
        for size in sizes:
            for seed in seeds:
                if (budget_proj, size, seed) in calculated:
                    continue
                if sample_method == 'pmedian':
                    selected_pairs = samples[size][seed][0]
                else:
                    selected_pairs = samples[size][seed]
                neighbors = find_neighbors(feature, selected_pairs, k=1)
                loss_bound = 0.3 * len(selected_pairs)
                if solving_method in ['naive', 'knn']:
                    args_new = gen_argument(args, selected_pairs, neighbors)
                    solver = BendersSolverOptimalityCutVariant(ins_name=ins_name, save_model=False)
                    weighted = True if solving_method == 'knn' else False
                    new_projects, new_signals, t, _ = solver.solve(args_new, budget_proj=budget_proj, budget_sig=0, beta_1=0.001, regenerate=True, pareto=True, relax4cut=True,
                                                                    weighted=weighted, quiet=True, manual_params=manual_params)
                elif solving_method == 'reg':
                    args_new = gen_argument(args, selected_pairs, neighbors, feature=feature)
                    solver = BendersRegressionSolverVariant(ins_name=ins_name, save_model=False)
                    new_projects, new_signals, t, loss_bound = solver.solve(args_new, budget_proj=budget_proj, budget_sig=0, beta_1=0.001, regenerate=True, pareto=True, relax4cut=True,
                                                                weighted=False, quiet=True, insample_weight=1, reg_factor=32, loss_bound=loss_bound, manual_params=manual_params)
                else:
                    raise ValueError('solution method {} is not defined'.format(solving_method))
                obj, acc, _ = cal_con_obj(args, new_projects, new_signals, manual_params=manual_params, beta=[])
                print('{} + {} (size: {}, seed: {}) -- obj: {:.4f} -- acc: {:.4f} -- time: {:.4f} sec'.format(sample_method, solving_method, size, seed, obj, acc, t))
                record = [budget_proj, size, seed, obj, acc, t, loss_bound, new_projects]
                records.append(record.copy())
                df = pd.DataFrame(records, columns=['budget', 'size', 'seed', 'obj', 'acc', 'time', 'loss_bound', 'projects'])
                dump_file(df_path, df)
                df.to_csv('./prob/{}/res/complete_{}/complete_{}_{}.csv'.format(ins_name, variant_type, sample_method, solving_method), index=False)


def complete_eval_utility(width, n_orig, budgets, sizes, sample_method='uniform', solving_method='naive', dim=16):
    # set parameters
    P, n_scenarios = 30, 10000
    alpha = 1.02
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}_n{}'.format(P, n_scenarios)
    seeds = [0, 12, 23, 34, 45, 56, 67, 78, 89, 90]
    # load data
    # instance
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70,
                                     p_sig=0.3, p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate_wroutes(save=True)
    # feature
    DW = DeepWalk_utility(random_seed=12, save=True)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=[], walk_per_node=50, walk_length=20, dim=dim)
    # std dict
    SSR = ScenarioSamplerRoute(P=P, save=False)
    utility_matrix = SSR.sample_train(args=args, n=n_scenarios, ins_name=ins_name)
    std_dict = cal_od_stds(utility_matrix, des2od(args['destinations']))
    # initialize the samples
    samples, _ = load_file('./prob/{}/samples/utility/active16_samples_{}.pkl'.format(ins_name, sample_method))
    # start the computation
    df_path = './prob/{}/res/complete_utility/complete_{}_{}.pkl'.format(ins_name, sample_method, solving_method)
    df, load_succeed = load_file(df_path)
    if load_succeed:
        calculated = {(budget, size, seed) for budget, size, seed in df[['budget', 'size', 'seed']].values}
        records = df.values.tolist()
    else:
        calculated = {}
        records = []
    for budget_proj in budgets:
        for size in sizes:
            for seed in seeds:
                if (budget_proj, size, seed) in calculated:
                    continue
                if sample_method == 'pmedian':
                    selected_pairs = samples[size][seed][0]
                else:
                    selected_pairs = samples[size][seed]
                neighbors = find_neighbors(feature, selected_pairs, k=1)
                loss_bound = 0.3 * len(selected_pairs)
                solver = BilevelSolver(ins_name, True)
                if solving_method == 'naive':
                    args_new = gen_argument(args, selected_pairs, neighbors)
                    _, t, new_projects = solver.solve(args_new, budget_proj, weighted=True, n_breakpoints=5)
                elif solving_method == 'knn':
                    neighbors = find_neighbors(feature, selected_pairs, k=1)
                    args_new = gen_argument_prod(args, selected_pairs, neighbors, std_dict)
                    _, t, new_projects = solver.solve(args_new, budget_proj, weighted=True, n_breakpoints=5)
                elif solving_method == 'reg':
                    c_feature_dict, o_feature_dict = gen_feature_dict(feature=feature, coreset=selected_pairs, od_pairs=des2od(args['destinations']))
                    args_new = gen_argument(args, selected_pairs, [])
                    _, t, new_projects = solver.pwo_solve(args_new, budget_proj, c_feature_dict, o_feature_dict, n_breakpoints=5, lambd=1, L_bar=loss_bound)
                else:
                    raise ValueError('solution method {} is not defined'.format(solving_method))
                obj = cal_utility_obj(args=args, new_projects=new_projects, alpha=alpha)
                print('{} + {} (size: {}, seed: {}) -- obj: {:.4f} -- time: {:.4f} sec'.format(
                    sample_method, solving_method, size, seed, obj, t))
                record = [budget_proj, size, seed, obj, t, loss_bound, new_projects]
                records.append(record.copy())
                df = pd.DataFrame(records, columns=['budget', 'size', 'seed', 'obj', 'time', 'loss_bound', 'projects'])
                dump_file(df_path, df)
                df.to_csv('./prob/{}/res/complete_utility/complete_{}_{}.csv'.format(ins_name, sample_method, solving_method), index=False)


def complete_eval_variants(variant_type='exp'):
    budgets = [100, 300, 500]
    sizes = [0.01, 0.02, 0.03, 0.04, 0.05]

    if variant_type == 'utility':
        # complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pmedian', solving_method='knn')
        # complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='uniform', solving_method='naive')
        # complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pmedian', solving_method='naive')
        # complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='uniform', solving_method='knn')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pcenter', solving_method='knn')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pcenter', solving_method='naive')
        # complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='uniform', solving_method='reg')
        # complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pmedian', solving_method='reg')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pcenter', solving_method='reg')
    else:
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='pcenter', solving_method='knn')
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='pcenter', solving_method='naive')
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='pcenter', solving_method='reg')
        # complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
        #                       variant_type=variant_type, sample_method='pmedian', solving_method='knn')
        # complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
        #                       variant_type=variant_type, sample_method='uniform', solving_method='naive')
        # complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
        #                       variant_type=variant_type, sample_method='pmedian', solving_method='naive')
        # complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
        #                       variant_type=variant_type, sample_method='uniform', solving_method='knn')
        # complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
        #                       variant_type=variant_type, sample_method='uniform', solving_method='reg')
        # complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
        #                       variant_type=variant_type, sample_method='pmedian', solving_method='reg')


def test_regression_approx(width=6, n_orig=36):
    param_dict = {(6, 36): (30, 10, 5000), (6, 72): (30, 10, 5000)}
    # set parameters
    P, U, n_scenarios = param_dict[width, n_orig]
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
    seeds = [0, 12, 23, 34, 45, 56, 67, 78, 89, 90]
    save = False
    # load data
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3,
                                     p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    DW = DeepWalk(random_seed=12, save=save)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=[], walk_per_node=50, walk_length=20, dim=32)
    # initialize the samples
    samples = gen_samples(width=width, n_orig=n_orig, feature=feature, args=args, sizes=[], seeds=seeds)
    # start the computation
    selected_pairs = samples['pmedian'][0.03][12]
    neighbors = find_neighbors(feature, selected_pairs, k=1)
    args_new = gen_argument(args, selected_pairs, neighbors, feature=feature)
    solver = BendersRegressionSolver(ins_name=ins_name, save_model=False)
    new_projects, new_signals, t = solver.solve(args_new, budget_proj=100, budget_sig=0,
                                                beta_1=0.001, regenerate=True, pareto=True, relax4cut=True,
                                                weighted=False, quiet=False, insample_weight=1, reg_factor=32, loss_bound=5)
    obj, acc, _ = cal_con_obj(args, new_projects, new_signals,
                              solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'], M=args['travel_time_max']))
    print('obj: {:.4f} -- acc: {:.4f} -- time: {:.4f} sec'.format(obj, acc, t))


def test_bagging_approx(width=6, n_orig=36):
    param_dict = {(6, 36): (30, 10, 5000)}
    # set parameters
    P, U, n_scenarios = param_dict[width, n_orig]
    ins_name = '{}x{}-{}'.format(width, width, n_orig)
    suffix = 'p{}-u{}-n{}'.format(P, U, n_scenarios)
    seeds = [0, 12, 23, 34, 45, 56, 67, 78, 89, 90]
    save = False
    # load data
    Generator = ClusterGridGenerator(width=width, n_orig=n_orig, discrete=False, time_limit=60, time_max=70, p_sig=0.3,
                                     p_orig_inter=0.7, n_inter=3, random_seed=12)
    args = Generator.generate(save=True)
    DW = DeepWalk(random_seed=12, save=save)
    feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=[], walk_per_node=50, walk_length=20, dim=32)
    # initialize the samples
    samples = gen_samples(width=width, n_orig=n_orig, feature=feature, args=args, sizes=[], seeds=seeds)
    # start the computation
    selected_pairs = samples['pmedian'][0.03][12]
    neighbors = find_neighbors(feature, selected_pairs, k=1)
    args_new = gen_argument(args, selected_pairs, neighbors, feature=feature)
    solver = BendersBaggingSolver(ins_name=ins_name, save_model=False)
    new_projects, new_signals, t, _ = solver.solve(args_new, budget_proj=100, budget_sig=0,
                                                beta_1=0.001, regenerate=True, pareto=True, relax4cut=True,
                                                weighted=False, quiet=False, knn_weight=0.5, reg_factor=32, loss_bound=5)
    obj, acc, _ = cal_con_obj(args, new_projects, new_signals,
                              solver._gen_betas(beta_1=0.001, T=args['travel_time_limit'], M=args['travel_time_max']))
    print('obj: {:.4f} -- acc: {:.4f} -- time: {:.4f} sec'.format(obj, acc, t))


if __name__ == '__main__':
    # visualization
    # instance_visual(width=6, n_orig=72)
    # initialize problems
    # prob_initialization(width=6, n_orig=72, dim=16)
    # prob_variant_initialization(variant_type='exp')
    # prob_variant_initialization(variant_type='linear')
    # prob_variant_initialization(variant_type='rec')
    # prob_utility_initialization(width=6, n_orig=72, dim=16)

    # training for n_sim analysis
    for v_type in ['utility']:
        for n_scenarios in [10, 100, 1000]:
            embedding_initialization_varaint_nsim_impact(width=6, n_orig=72, dim=16, variant_type=v_type, n_scenarios=n_scenarios)

    # training for dimensionality analysis
    # for dim in [2, 4, 8, 16, 32]:
    #     prob_variant_initialization(variant_type='exp', dim=dim)
    #     prob_variant_initialization(variant_type='linear', dim=dim)
    #     prob_variant_initialization(variant_type='rec', dim=dim)
    #     prob_utility_initialization(width=6, n_orig=72, dim=dim)

    # budget_impact(budget_scenarios=[(50, 0), (100, 0), (150, 0), (200, 0), (250, 0), (300, 0)], n_repeat=10)
    # sample_size_impact(budget_proj=300, budget_sig=0, n_repeat=10,
    #                    sizes=[0.01, 0.03, 0.05, 0.08, 0.10, 0.15, 0.20, 0.25, 0.30])
    # embedding_performance_fixed_size(budget_scenarios=[(50, 0), (100, 0), (150, 0), (200, 0), (250, 0), (300, 0)], n_repeat=10)
    # embedding_performance_fixed_budget(budget_proj=200, budget_sig=0, n_repeat=10, sizes=sizes)
    # opt_sol([(50, 0), (100, 0), (150, 0), (200, 0), (250, 0), (300, 0), (400, 0), (500, 0), (700, 0), (1000, 0)])

    # complete evaluation
    # complete_eval(width=6, n_orig=72, budgets=[100, 300, 500], sizes=[0.01, 0.02, 0.03, 0.04, 0.05],
    #               sample_method='uniform', solving_method='reg', dim=16)
    # complete_eval_variants(variant_type='exp')
    # complete_eval_variants(variant_type='linear')
    # complete_eval_variants(variant_type='rec')
    # complete_eval_variants(variant_type='utility')

    # some tests
    # test_regression_approx()
    # test_bagging_approx()
