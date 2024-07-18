import yaml
import preprocess
import train
import test
import argparse
import os
from common import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_version', type=int, help='Data version', default=None)
    args = parser.parse_args()

    data_config = get_config('data')
    if data_config['data_version'] == 'latest':
        try:
            latest_version = 1 + max([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(data_config['package_path'] + 'data/processed/') if 'joint_state_data' in f])
            data_config['data_version'] = latest_version
        except:
            data_config['data_version'] = 1
    preprocess.run(data_config)

    train_config = get_config('train')
    train_config['data_version'] = data_config['data_version']
    train.run(train_config)
    
    test_config = get_config('test')
    if args.test_data_version is None:
        test_config['data_version'] = data_config['data_version']
    else:
        test_config['data_version'] = args.test_data_version
    test_config['exp_name'] = train_config['exp_name']
    test.run(test_config)

if __name__ == '__main__':
    main()