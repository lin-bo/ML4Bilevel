from embedding.embedding_gpu import build_corpus, deep_walk_training
import argparse


if __name__ == '__main__':
    # set parser
    parser = argparse.ArgumentParser()
    parser.add_argument('--nworkers', type=int, help='number of cpus')
    parser.add_argument('--suffix', type=str)
    parser.add_argument('--weight_file_name', type=str)
    args = parser.parse_args()
    # build corpus
    build_corpus(ins_name='trt', walk_per_node=50, walk_length=50, n_workers=args.nworkers, weight_file_name=args.weight_file_name)
    # deep walk training
    _, _ = deep_walk_training(dim=32, n_workers=args.nworkers, suffix=args.suffix)
