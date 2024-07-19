import torch
import numpy as np
import preprocess
import train
import test
import argparse
import os
from common import *

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--test_data_version', type=int, help='Data version', default=None)
    parser.add_argument('--use_existing_data', type=int, help='Use old data', default=False)
    args = parser.parse_args()

    data_config = get_config('data')
    if data_config['data_version'] == 'latest':
        try:
            latest_version = max([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(data_config['package_path'] + 'data/processed/') if 'state_data' in f])
        except:
            latest_version = 1
    if args.use_existing_data:
        data_config['data_version'] = latest_version
    else:
        data_config['data_version'] = latest_version + 1
        preprocess.run(data_config)
    print(f'Data version: {data_config["data_version"]}')

    run_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f'Run name: {run_name}')
    os.makedirs(data_config['package_path'] + f'logs/{run_name}', exist_ok=True)
    os.makedirs(data_config['package_path'] + f'logs/latest', exist_ok=True)
    with open(data_config['package_path'] + f'logs/{run_name}/data.yaml', 'w') as f:
        yaml.dump(data_config, f)


    train_config = get_config('train')
    train_config['data_version'] = data_config['data_version']
    train_config['run_name'] = run_name
    
    torch.manual_seed(train_config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(train_config['seed'])

    with open(data_config['package_path'] + f'logs/{run_name}/train.yaml', 'w') as f:
        yaml.dump(train_config, f)
    train.run(train_config)

    test_config = get_config('test')
    test_config['run_name'] = run_name
    if args.test_data_version is None:
        test_config['data_version'] = data_config['data_version']
    else:
        test_config['data_version'] = args.test_data_version
    test.run(test_config)

if __name__ == '__main__':
    main()