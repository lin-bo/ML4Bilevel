from utils.instance_generator import ClusterGridGenerator
from embedding.embedding import DeepWalk_variant, DeepWalk_utility, ScenarioSamplerRoute
from embedding.coreset import gen_argument, gen_argument_prod, find_neighbors, gen_feature_dict
from solver.continuous.benders import BendersSolverOptimalityCutVariant, BendersRegressionSolverVariant
from solver.route_choice.bilevel import BilevelSolver
from utils.functions import cal_con_obj, cal_utility_obj, des2od, dump_file, load_file, cal_od_stds
import pandas as pd
import argparse


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
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pmedian', solving_method='knn')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='uniform', solving_method='naive')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pmedian', solving_method='naive')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='uniform', solving_method='knn')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pcenter', solving_method='knn')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pcenter', solving_method='naive')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='uniform', solving_method='reg')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pmedian', solving_method='reg')
        complete_eval_utility(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16, sample_method='pcenter', solving_method='reg')
    else:
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='pcenter', solving_method='knn')
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='pcenter', solving_method='naive')
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='pcenter', solving_method='reg')
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='pmedian', solving_method='knn')
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='uniform', solving_method='naive')
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='pmedian', solving_method='naive')
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='uniform', solving_method='knn')
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='uniform', solving_method='reg')
        complete_eval_variant(width=6, n_orig=72, budgets=budgets, sizes=sizes, dim=16,
                              variant_type=variant_type, sample_method='pmedian', solving_method='reg')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, help='name of the accessibility measure: exp, linear, rec, utility')
    args = parser.parse_args()
    # complete evaluation
    complete_eval_variants(variant_type=args.variant)
