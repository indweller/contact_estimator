import yaml
import preprocess
import train
import test
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description='Process some integers.')
    package_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    package_path = '/'.join(package_path.split('/')[:-2]) + '/'
    parser.add_argument('--absolute_package_path', type=str, help='Absolute path to package', default=package_path)
    parser.add_argument('--test_data_version', type=int, help='Data version', default=1)
    args = parser.parse_args()

    data_config = yaml.load(open(args.absolute_package_path + 'config/data.yaml', 'r'), Loader=yaml.FullLoader)
    data_config['package_path'] = args.absolute_package_path
    if data_config['data_version'] == 'latest':
        try:
            latest_version = 1 + max([int(f.split('_')[-1].split('.')[0]) for f in os.listdir(data_config['package_path'] + 'data/processed/') if 'joint_state_data' in f])
            data_config['data_version'] = latest_version
        except:
            data_config['data_version'] = 1
    preprocess.run(data_config)
    train_config = yaml.load(open(args.absolute_package_path + 'config/train.yaml', 'r'), Loader=yaml.FullLoader)
    train_config['package_path'] = args.absolute_package_path
    train_config['data_version'] = data_config['data_version']
    train.run(train_config)
    test_config = yaml.load(open(args.absolute_package_path + 'config/test.yaml', 'r'), Loader=yaml.FullLoader)
    test_config['package_path'] = args.absolute_package_path
    test_config['data_version'] = args.test_data_version
    test_config['exp_name'] = train_config['exp_name']
    test.run(test_config)

if __name__ == '__main__':
    main()