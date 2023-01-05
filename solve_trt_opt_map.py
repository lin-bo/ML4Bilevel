#!/usr/bin/env python
# coding: utf-8
# Author: Bo Lin

import pandas as pd
from utils.functions import load_file, str2list
import geopandas as gpd
import argparse


def extract_projects(proj2art, projects):
    arts = []
    for p in projects:
        arts += proj2art[p]
    return arts


def gen_proj_shp(n, sn, budget, potential, region=None):
    # set sub-region suffix
    if region == None:
        suffix = ''
    else:
        suffix = '_{}'.format(region)
    filename = 'efficiency-n{}_id{}_summary'.format(n, sn)
    budget_col = 'budget'
    project_col = 'projects'
    df = pd.read_csv('./prob/trt/res{}/{}/summary/{}.csv'.format(suffix, potential, filename))
    projects = df[df[budget_col] <= budget][project_col].values[-1]
    print('Projects: ', projects)
    df_art = gpd.read_file('./data/trt_arterial/trt_arterial.shp')
    proj2artidx, _ = load_file('./data/trt_instance/proj2artid{}.pkl'.format(suffix))
    art_index = extract_projects(proj2artidx, str2list(projects))
    df_selected = df_art.loc[art_index, :].copy()
    df_selected.to_file(driver='ESRI Shapefile', filename='./prob/trt/res{}/shp/{}-budget{}.shp'.format(suffix, filename, budget))


if __name__ == '__main__':
    # parser
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--sns', nargs='+', type=int, help='series number of the p-median samples in a list')
    parser.add_argument('--n', type=int, help='number of od pairs in the sample')
    parser.add_argument('-b', '--budget', type=int, help='budget (km x 4)')
    parser.add_argument('--potential', type=str, help='the potential for accessibility calculation, job/populations')
    parser.add_argument('--region', type=str, help='region name to check for in folder')
    args = parser.parse_args()
    # generate shapefile
    for sn in args.sns:
        gen_proj_shp(n=args.n, sn=sn, budget=args.budget, potential=args.potential, region=args.region)
