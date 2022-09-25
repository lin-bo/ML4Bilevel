#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

import pandas as pd
from utils.functions import load_file, str2list, cal_acc
import os
import argparse


def sol_summary(potential='job',  n=500, sn=10):
    # load data
    filename = 'efficiency-n{}_id{}'.format(n, sn)
    if os.path.exists('./prob/trt/res/{}/{}.csv'.format(potential, filename)):
        res = pd.read_csv('./prob/trt/res/{}/{}.csv'.format(potential, filename))
    else:
        return None
    sum_path = './prob/trt/res/{}/summary/{}_summary.csv'.format(potential, filename)
    if os.path.exists(sum_path):
        df = pd.read_csv(sum_path)
        calculated = {b for b in df['budget'].values}
        records = df.values.tolist()
    else:
        records, calculated = [], {}
    args, _ = load_file('./prob/trt/args.pkl')
    projs, budgets = res['projects'].values, res['budget_proj'].values
    for idx, p in enumerate(projs):
        if budgets[idx] in calculated:
            continue
        proj = str2list(p)
        acc = cal_acc(args, proj, [], impedence='time', potential=potential)
        records.append([budgets[idx], acc, p])
        df = pd.DataFrame(records, columns=['budget', '{}_acc'.format(potential), 'projects'])
        df.to_csv(sum_path, index=False)


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sns', nargs='+', type=int, help='series number of the p-median samples in a list')
    parser.add_argument('--n', type=int, help='number of od pairs in the sample')
    parser.add_argument('--potential', type=str, help='the potential for accessibility calculation, job/populations')
    args = parser.parse_args()
    # summarize results
    for sn in args.sns:
        sol_summary(potential=args.potential, n=args.n, sn=sn)
