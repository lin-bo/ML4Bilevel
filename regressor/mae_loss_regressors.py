import numpy as np
import gurobipy as gp

class MAELasso:
    def __init__(self, alpha=0.1, fit_intercept=True):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.w = []

    def fit(self, X, y):
        """
        fit the regression model
        :param X: (n x d) array
        :param y: (n,) array
        :return: None
        """
        if self.fit_intercept:
            X = self._add_intercept_col(X)
        model = self._construct_lp(X, y)
        model.optimize()
        self.w = self._get_coeff(model)

    def predict(self, X):
        """
        make predictions
        :param X: (n x d) array
        :return: (n,) array
        """
        if self.fit_intercept:
            X = self._add_intercept_col(X)
        return np.dot(X, self.w).reshape(-1)

    def _add_intercept_col(self, X):
        n, _ = X.shape
        return np.concatenate([np.ones((n, 1)), X], axis=1)

    def _construct_lp(self, X, y):
        # set parameters
        n, dim = X.shape
        N = list(range(n))
        D = list(range(dim))
        # initialize model
        model = gp.Model('mae_lasso')
        model.Params.outputFlag = 0
        # add variables
        w = model.addVars(D, name='w', ub=gp.GRB.INFINITY, lb=-gp.GRB.INFINITY, vtype=gp.GRB.CONTINUOUS)
        a = model.addVars(D, name='a', ub=gp.GRB.INFINITY, lb=0, vtype=gp.GRB.CONTINUOUS)
        e = model.addVars(N, name='e', ub=gp.GRB.INFINITY, lb=0, vtype=gp.GRB.CONTINUOUS)
        # add constraints
        model.addConstrs((e[i] >= gp.quicksum(w[d] * X[i, d] for d in D) - y[i] for i in N), name='error_pos')
        model.addConstrs((e[i] >= y[i] - gp.quicksum(w[d] * X[i, d] for d in D) for i in N), name='error_neg')
        model.addConstrs((a[d] >= w[d] for d in D), name='regu_pos')
        model.addConstrs((a[d] >= -w[d] for d in D), name='regu_neg')
        # set obj
        obj = e.sum() / n + self.alpha * gp.quicksum(a[d] for d in D[1:]) / (dim - 1)
        model.setObjective(obj, gp.GRB.MINIMIZE)
        # store variables
        model._w = w
        return model

    def _get_coeff(self, model):
        w_val = model.getAttr('x', model._w)
        coeff = np.array([w_val[i] for i in range(len(w_val))]).reshape((-1, 1))
        return coeff