import numpy as np
import pandas as pd
from utils.functions import load_file, dump_file, str2list, cal_acc, gen_betas, des2od
import os
import geopandas as gpd


def sol_summary(potential='job', method='greedy', n=500, sn=10, regu=0.):
    # load data
    if method == 'greedy':
        filename = 'greedy_abs_{}_par'.format(potential)
        # res = load_file('./prob/trt/res/{}/{}.pkl'.format(potential, filename))
    else:
        filename = 'efficiency-n{}_id{}'.format(n, sn)
    if regu > 0:
        filename += '_regu{}'.format(regu)
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
    if method == 'greedy':
        budgets = np.arange(40, 401, 40)
        projs = [res[res['allocated'] <= b]['selected'].values[-1] for b in budgets]
    else:
        projs, budgets = res['projects'].values, res['budget_proj'].values
    for idx, p in enumerate(projs):
        if budgets[idx] in calculated:
            continue
        proj = str2list(p)
        acc = cal_acc(args, proj, [], impedence='time', potential=potential)
        records.append([budgets[idx], acc, p])
        df = pd.DataFrame(records, columns=['budget', '{}_acc'.format(potential), 'projects'])
        df.to_csv(sum_path, index=False)


def extract_projects(proj2art, projects):
    arts = []
    for p in projects:
        arts += proj2art[p]
    return arts


def gen_proj_shp(n, sn, budget, potential, greedy=False):
    filename = 'efficiency-n{}_id{}'.format(n, sn) if not greedy else 'greedy_abs_{}_par'.format(potential)
    budget_col = 'budget_proj' if not greedy else 'allocated'
    project_col = 'projects' if not greedy else 'selected'
    df = pd.read_csv('./prob/trt/res/{}/{}.csv'.format(potential, filename))
    projects = df[df[budget_col] <= budget][project_col].values[-1]
    print('Projects: ', projects)
    df_art = gpd.read_file('./data/trt_arterial/trt_arterial.shp')
    proj2artidx, _ = load_file('./data/trt_instance/proj2artid.pkl')
    art_index = extract_projects(proj2artidx, str2list(projects))
    df_selected = df_art.loc[art_index, :].copy()
    df_selected.to_file(driver='ESRI Shapefile', filename='./prob/trt/res/shp/{}-budget{}.shp'.format(filename, budget))


def project_convert(selected_projects):
    args, _ = load_file('./prob/trt/args.pkl')
    projects = args['projects']
    edges = []
    for p in selected_projects:
        edges += projects[p]
    args, _ = load_file('./prob/trt/args_wo_yonge.pkl')
    projects = args['projects']
    return [idx for idx, p in enumerate(projects) if len([1 for f, t in edges if (f, t) in p])]


if __name__ == '__main__':
    # summarize results
    for sn in range(21, 42):
        sol_summary(potential='job', method='opt', n=2000, sn=sn)
    # generate shapefile
    gen_proj_shp(n=2000, sn=34, budget=280, potential='job', greedy=False)
