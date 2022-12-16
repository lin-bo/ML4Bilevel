from solver.continuous.greedy import GreedySolver, GreedySolverPar
from utils.instance_generator import RealInstanceGenerator
import argparse


def solve_greedy(budget, metric):
    # load data
    Generator = RealInstanceGenerator()
    args = Generator.generate()
    # initialize the solver
    solver = GreedySolverPar(metric=metric, n_workers=4)
    solver.solve(args, budget)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--budget', type=int, help='road design budget')
    args = parser.parse_args()
    solve_greedy(args.budget, 'abs')
