from solver.continuous.greedy import GreedySolver, GreedySolverPar
from utils.instance_generator import RealInstanceGenerator
import argparse


def solve_greedy(budget, metric, potential='job', region=None):
    # load data
    Generator = RealInstanceGenerator()
    args = Generator.generate(region=region)
    # initialize the solver
    solver = GreedySolverPar(metric=metric, n_workers=4, potential=potential, region=region)
    solver.solve(args, budget)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--budget', type=int, help='road design budget')
    parser.add_argument('--potential', type=str, help='type of accessibility', default='job')
    parser.add_argument('--region', type=str, help='region name', default=None)
    args = parser.parse_args()
    solve_greedy(args.budget, 'abs', args.potential, args.region)
