import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from utils.functions import load_file, des2od, time2acc_synthetic, flatten
import numpy as np
from embedding.embedding import DeepWalk_utility, DeepWalk_variant


class MLP(nn.Module):

    def __init__(self, INPUT_DIM):
        super().__init__()
        self.l1 = nn.Linear(INPUT_DIM, 64)
        self.l2 = nn.Linear(64, 1)

    def forward(self, x):
        x = F.relu(self.l1(x))
        x = self.l2(x)
        return x


class AccDatasetTrain(Dataset):

    def __init__(self, ins_name='6x6-72', vtype='exp', od_fea=True):
        super().__init__()
        # load data
        if vtype == 'utility':
            target_matrix, _ = load_file(f'./prob/{ins_name}/pred/fixed_budget_utilitymat_tr.pkl')
            target_matrix = np.array(target_matrix)
            std_vec = target_matrix.std(axis=0, keepdims=True)
            min_vec = target_matrix.min(axis=0, keepdims=True)
            target_matrix = (target_matrix / min_vec - 1) / std_vec
            leader_sol, _ = load_file(f'./prob/{ins_name}/pred/fixed_budget_uleadersol_tr.pkl')
            args, _ = load_file(f'./prob/{ins_name}/args_c.pkl')
            DW = DeepWalk_utility(random_seed=12, save=False)
        else:
            time_matrix, _ = load_file(f'./prob/{ins_name}/pred/fixed_budget_timemat_tr.pkl')
            leader_sol, _ = load_file(f'./prob/{ins_name}/pred/fixed_budget_leadersol_tr.pkl')
            target_matrix = time2acc_synthetic(np.array(time_matrix) * 6 / 7, variant_type=vtype)
            args, _ = load_file(f'./prob/{ins_name}/args_c.pkl')
            DW = DeepWalk_variant(random_seed=12, save=True, variant_type=vtype)
        # calculate the target variable
        pop = args['populations']
        od_pairs = des2od(args['destinations'])
        weight = np.array([pop[des] for orig, des in od_pairs]).reshape((1, -1))
        y = np.array(target_matrix * weight).sum(axis=1, keepdims=True)
        self.y = torch.tensor(y, dtype=torch.float32)
        # calculate the features
        suffix = 'p30_n10000' if vtype == 'utility' else'p25-u10-n5000'
        od_feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=[], walk_per_node=50, walk_length=20, dim=16)
        od_feature = od_feature.mean(axis=0)
        od_feature = np.array([od_feature for _ in range(len(leader_sol))])
        r_idx = [i for i, sol in enumerate(leader_sol) for _ in sol]
        c_idx = flatten(leader_sol)
        leader_feature = np.zeros((len(leader_sol), 84))
        leader_feature[r_idx, c_idx] = 1
        if od_fea:
            x = np.concatenate([leader_feature, od_feature], axis=1)
            self.x = torch.tensor(x, dtype=torch.float32)
        else:
            self.x = torch.tensor(leader_feature, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


class AccDatasetTest(Dataset):

    def __init__(self, ins_name='6x6-72', vtype='exp', budget=100, od_fea=True):
        super().__init__()
        # load data
        if vtype == 'utility':
            target_matrix, _ = load_file(f'./prob/{ins_name}/pred/fixed_budget_utilitymat_{budget}.pkl')
            target_matrix = np.array(target_matrix)
            std_vec = target_matrix.std(axis=0, keepdims=True)
            min_vec = target_matrix.min(axis=0, keepdims=True)
            target_matrix = (target_matrix / min_vec - 1) / std_vec
            leader_sol, _ = load_file(f'./prob/{ins_name}/pred/fixed_budget_uleadersol_{budget}.pkl')
            args, _ = load_file(f'./prob/{ins_name}/args_r.pkl')
            DW = DeepWalk_utility(random_seed=12, save=False)
        else:
            time_matrix, _ = load_file(f'./prob/{ins_name}/pred/fixed_budget_timemat_{budget}.pkl')
            leader_sol, _ = load_file(f'./prob/{ins_name}/pred/fixed_budget_leadersol_{budget}.pkl')
            target_matrix = time2acc_synthetic(np.array(time_matrix) * 6 / 7, variant_type=vtype)
            args, _ = load_file(f'./prob/{ins_name}/args_c.pkl')
            DW = DeepWalk_variant(random_seed=12, save=True, variant_type=vtype)
        # calculate the target variable
        pop = args['populations']
        od_pairs = des2od(args['destinations'])
        weight = np.array([pop[des] for orig, des in od_pairs]).reshape((1, -1))

        y = np.array(target_matrix * weight).sum(axis=1, keepdims=True)
        self.y = torch.tensor(y, dtype=torch.float32)
        # calculate the features
        suffix = 'p30_n10000' if vtype == 'utility' else'p25-u10-n5000'
        od_feature = DW.node2vec(ins_name=ins_name, suffix=suffix, weights=[], walk_per_node=50, walk_length=20, dim=16)
        od_feature = od_feature.mean(axis=0)
        od_feature = np.array([od_feature for _ in range(len(leader_sol))])
        r_idx = [i for i, sol in enumerate(leader_sol) for _ in sol]
        c_idx = flatten(leader_sol)
        leader_feature = np.zeros((len(leader_sol), 84))
        leader_feature[r_idx, c_idx] = 1
        if od_fea:
            x = np.concatenate([leader_feature, od_feature], axis=1)
            self.x = torch.tensor(x, dtype=torch.float32)
        else:
            self.x = torch.tensor(leader_feature, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


def train_mlp(net, optimizer, epoch, train_loader, device):
    net.train()
    total_loss = 0.
    for x, y in train_loader:
        x, y = x.to(device), y.to(device)
        net.zero_grad()
        pred = net.forward(x)
        loss = F.mse_loss(pred, y, reduction='sum')
        loss.backward()
        optimizer.step()
        total_loss += loss
    if epoch % 100 == 0:
        print(f'Epoch: {epoch}, loss: {total_loss:.0f}')


def test_mlp(net, test_loader, device, ins_name, v_type, budget, od_fea):
    total_loss = 0.0
    ml_loss = []
    for x, y in test_loader:
        x, y = x.to(device), y.to(device)
        pred = net.forward(x)
        ml_loss += torch.abs(pred - y).reshape(-1).tolist()
        total_loss += torch.abs(pred - y).sum()
    print(f'test MAE: {total_loss/1000}')
    df = pd.DataFrame({'idx': range(len(ml_loss)), 'mae': ml_loss})
    if od_fea:
        df.to_csv(f'./prob/{ins_name}/pred/{v_type}/ml_loss_{budget}_w.csv', index=False)
    else:
        df.to_csv(f'./prob/{ins_name}/pred/{v_type}/ml_loss_{budget}_wo.csv', index=False)
