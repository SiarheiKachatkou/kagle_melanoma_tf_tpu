import argparse

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--work_dir', type=str)
    parser.add_argument('--folds', type=int, default=0)

    args = parser.parse_args()
    return args