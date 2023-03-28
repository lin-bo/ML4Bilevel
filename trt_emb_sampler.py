from utils.instance_generator import RealInstanceGenerator
from embedding.embedding import ScenarioSampler
import argparse


def sample_scenarios(P, U, n_workers, n):
    # set params
    ins_name = 'trt'
    # load instance
    Generator = RealInstanceGenerator()
    args = Generator.generate()
    # sampling scenarios
    SS = ScenarioSampler(P=P, U=U, save=True, multiproc=True, n_workers=n_workers, weight='time')
    SS.sample_train(args=args, ins_name=ins_name, n=n)


if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_workers', type=int, help='number of cpus available')
    parser.add_argument('-p', '--num_projects', type=int, help='upper bound on the number of projects to select in each scenarios')
    parser.add_argument('-u', '--num_signals', type=int, help='upper bound on the number of signals to select in each scenarios')
    parser.add_argument('-n', '--num_scenarios', type=int, help='number of scenarios (network design decisions) to sample')
    args = parser.parse_args()
    # sample scenarios
    sample_scenarios(P=args.num_projects, U=args.num_signals, n_workers=args.n_workers, n=args.num_scenarios)
