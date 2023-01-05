from solver.continuous.greedy import GreedySolver, GreedySolverPar
from utils.instance_generator import RealInstanceGenerator
import argparse


def solve_greedy(budget, metric, region):
    # load data
    Generator = RealInstanceGenerator()
    args = Generator.generate(region=region)
    # initialize the solver
    solver = GreedySolverPar(metric=metric, n_workers=4)
    solver.solve(args, budget, region)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--budget', type=int, help='road design budget')
    parser.add_argument('--region', type=str, help='region name')
    args = parser.parse_args()
    solve_greedy(args.budget, 'abs', args.region)
