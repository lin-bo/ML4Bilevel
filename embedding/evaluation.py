import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression, Lasso
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, recall_score, precision_score
from utils.functions import gen_betas
from scipy.spatial import distance_matrix
from regressor.mae_loss_regressors import MAELasso
import matplotlib.pyplot as plt


def search_best_const(y, n=100):
    y_min, y_max = np.min(y), np.max(y)
    step = (y_max - y_min)/(n-1)
    candidates = [y_min + i * step for i in range(n)]
    best_conts = y_max
    best_abs = 1e5
    length = len(y)
    for c in candidates:
        abs = mean_absolute_error(np.ones(length) * c, y)
        if abs < best_abs:
            best_abs = abs
            best_conts = c
    return best_conts


def run_test(x, y, task, model_name, test_size, train_set):
    # select a model to fit
    models = {'clf': {'knn': KNeighborsClassifier, 'logistic': LogisticRegression},
              'reg': {'knn': KNeighborsRegressor, 'lr': LinearRegression, 'lasso': Lasso, 'mlp': MLPRegressor,
                      'mae_lasso': MAELasso}}
    params = {'knn': 2, 'lasso': 0.01, 'mlp': [4], 'mae_lasso': 2.5}
    naive_params = {'knn': 1, 'lasso': 0.01, 'mlp': [4], 'mae_lasso': 0.1}
    if (task in models) and (model_name in models[task]):
        model = models[task][model_name](params[model_name]) if model_name in params else \
            models[task][model_name]()
        model_naive = models[task][model_name](naive_params[model_name]) if model_name in naive_params else \
            models[task][model_name]()
    else:
        raise ValueError('The given task or model is not found')

    # prepare the data
    n = int(len(y) * 0.5)
    if len(train_set) > 0:
        x_train, y_train = x[train_set], y[train_set]
        flag = np.ones(len(x), dtype=bool)
        flag[train_set] = False
        x_test, y_test = x[flag], y[flag]
    else:
        if test_size > 1:
            test_size = test_size / len(y)
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size)
    # fit and predict
    model.fit(X=x_train, y=y_train)
    pred_train, pred_test = model.predict(x_train), model.predict(x_test)
    # print(pred_test)
    if task == 'reg':
        score_train = mean_absolute_error(pred_train, y_train)
        score_test = mean_absolute_error(pred_test, y_test)
        model_naive.fit(X=x_train[:, [0, 1]], y=y_train)
        # naive_score_train = mean_absolute_error(model_naive.predict(x_train[:, [0, 1]]), y_train)
        # naive_score_test = mean_absolute_error(model_naive.predict(x_test[:, [0, 1]]), y_test)
        # plt.scatter(y_test, pred_test, alpha=0.1)
        # plt.show()
        y_mean = search_best_const(y_train, n=100)
        naive_score_train = mean_absolute_error(np.ones(len(y_train)) * y_mean, y_train)
        naive_score_test = mean_absolute_error(np.ones(len(y_test)) * y_mean, y_test)
        return [score_train, score_test, naive_score_train, naive_score_test]
    elif task == 'clf':
        score_train = model.score(x_train, y_train)
        score_test = model.score(x_test, y_test)
        naive_score_train = 1 - np.sum(y_train) / len(y_train)
        naive_score_test = 1 - np.sum(y_test) / len(y_test)
        # recall_train = recall_score(y_true=y_train, y_pred=pred_train) if pred_train.sum() > 0 else np.nan
        # recall_test = recall_score(y_true=y_test, y_pred=pred_test) if pred_test.sum() > 0 else np.nan
        # precision_train = precision_score(y_true=y_train, y_pred=pred_train) if pred_train.sum() > 0 else np.nan
        # precision_test = precision_score(y_true=y_test, y_pred=pred_test) if pred_test.sum() > 0 else np.nan
        return [score_train, score_test, naive_score_train, naive_score_test]


def emb_eval(feature, target, test_size=0.9, size=500, task='clf', model_name='logistic', train_set=[], datatype='01'):
    n_test, n_pair = target.shape
    if datatype == 'utility':
        test_scenarios = list(range(n_test))
    else:
        flag = (target.sum(axis=1) < target.shape[1]) * (target.sum(axis=1) > 0)
        test_scenarios = [idx for idx in range(n_test) if flag[idx]][:size]
    records = []
    print('# pairs: {}, core set: {}, # scenarios: {}/{}'.format(n_pair, len(train_set), len(test_scenarios), n_test))
    for s in test_scenarios:
        record = run_test(x=feature, y=target[s], task=task, model_name=model_name, test_size=test_size, train_set=train_set)
        print(s, record)
        records.append(record)
    # print(len(records), len(records[0]))
    df_test = pd.DataFrame(records, columns=['score_train', 'score_test', 'naive_score_train', 'naive_score_test'])  # 'recall_train', 'recall_test', 'precision_train', 'precision_test'
    df_test.dropna(axis=0, inplace=True)
    return df_test


def time2penalty(time_matrix, M, T, beta_1):
    beta = gen_betas(beta_1, T, M)
    penalty = (time_matrix < 0).astype(float) * beta[0]
    penalty += (time_matrix >= 0) * time_matrix * beta[1]
    penalty += (time_matrix > T) * (time_matrix - T) * (beta[2] - beta[1])
    return penalty


def nearest_pairs(feature, pair_idx, od_pairs, k=3):
    vec = feature[pair_idx]
    dist = distance_matrix([vec], feature)[0]
    dist_idx = np.argsort(dist)
    nn_idx = dist_idx[1:k+1]
    nn = [od_pairs[i] for i in nn_idx]
    # print('emb: {}-nn of pair {}:'.format(k, od_pairs[pair_idx]), nn)
    return nn_idx