import sys
import os
import shutil
import argparse
import numpy as np


def parse_args(args):
    parser = argparse.ArgumentParser(description='Split data into train and test sets.')
    parser.add_argument('subjects_root_path', type=str, help='Directory containing subject sub-directories.')
    parser.add_argument('--seed', type=int, default=100, help='random seed')
    parser.add_argument('--test-proportion', type=float, default=0.25, help='random seed')
    parser.add_argument(
            '--train-csv',
            type=str,
            default="../../data/mimic/train_ids.csv")
    parser.add_argument(
            '--test-csv',
            type=str,
            default="../../data/mimic/test_ids.csv")
    args, _ = parser.parse_known_args(args)
    return args

def main(args=sys.argv[1:]):
    args = parse_args(args)
    np.random.seed(args.seed)


    folders = os.listdir(args.subjects_root_path)
    patient_ids = list(filter(str.isdigit, folders))
    num_test = int(args.test_proportion * len(patient_ids))

    shuffled_patient_ids = np.random.permutation(patient_ids)
    train_patients = np.array(shuffled_patient_ids[:-num_test], dtype=int).reshape((-1,1))
    test_patients = np.array(shuffled_patient_ids[-num_test:], dtype=int).reshape((-1,1))
    print(test_patients)

    np.savetxt(args.train_csv, train_patients, fmt='%d', delimiter=",")
    np.savetxt(args.test_csv, test_patients, fmt='%d', delimiter=",")

if __name__ == '__main__':
    main()
