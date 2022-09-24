#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

from utils.functions import load_file, str2list
from embedding.coreset import gen_argument, find_neighbors
from solver.continuous.benders import BendersSolverOptimalityCut, BendersSolverEquity
import pandas as pd
import os
import argparse


def solve_trt(sn, n_sample, budgets, potential='job'):
    # set params
    ins_name = 'trt'
    budget_sig = 0
    # load data
    print('loading data ...')
    args, _ = load_file('./prob/trt/args_adj_ratio.pkl')
    feature, _ = load_file('./prob/trt/emb/emb_ratio.pkl')
    selected_pairs = str2list(pd.read_csv('./prob/trt/pmedian/sample{}_p{}.csv'.format(sn, n_sample))['medians'].values[-1])
    # find neighbors
    print('searching for neighbors ...')
    neighbors = find_neighbors(feature, selected_pairs, k=1)
    args_new = gen_argument(args, selected_pairs, neighbors, potential=potential)
    # running exp
    print('computation starts ...')
    path = './prob/{}/res/{}/efficiency-n{}_id{}.csv'.format(ins_name, potential, n_sample, sn)
    print('\nVarying budget in ', budgets)
    if os.path.exists(path):
        vals = pd.read_csv(path).values.tolist()
        records = [[val[0], str2list(val[1]), val[-1]] for val in vals]
        calculated = {r[0] for r in records}
    else:
        records = []
        calculated = {}
    for budget_proj in budgets:
        if budget_proj in calculated:
            continue
        print('\n*******************')
        print('budget:', budget_proj)
        print('*******************\n')
        solver = BendersSolverOptimalityCut(ins_name=ins_name, save_model=False, potential=potential)
        new_projects, _, t_uniform, _ = solver.solve(args_new, budget_proj=budget_proj, budget_sig=budget_sig, beta_1=0.001,
                                                     regenerate=True, pareto=True, relax4cut=True, weighted=True, quiet=False,
                                                     time_limit=10800)
        records.append([budget_proj, new_projects, t_uniform])
        df = pd.DataFrame(records, columns=['budget_proj', 'projects', 'time'])
        df.to_csv(path, index=False)


if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sns', nargs='+', type=int, help='series number of the p-median samples in a list')
    parser.add_argument('-b', '--budgets', nargs='+', type=int, help='budgets that we want to consider in a list')
    parser.add_argument('--n', type=int, help='number of od pairs in the sample')
    parser.add_argument('--potential', type=str, help='the potential for accessibility calculation, job/populations')
    args = parser.parse_args()
    # run experiment
    for sn in args.sns:
        solve_trt(sn, args.n, args.budgets, potential=args.potential)
