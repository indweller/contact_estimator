import torch
import torch.nn as nn
import numpy as np
import yaml
import os

import data_utils
import models

def get_accuracy_CE(outputs, labels):
    return (outputs.argmax(dim=1) == labels).float().mean()

def get_accuracy_BCE(outputs, labels):
    return ((outputs > 0).float() == labels).float().mean()

def test(model, test_loader, acc_fn):
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    acc = 0
    with torch.no_grad():
        for i, data in enumerate(test_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            acc += acc_fn(outputs, labels)
    return acc / len(test_loader)

def run(config):
    if config == None:
        config = yaml.safe_load(open('config/test.yaml'))
    run_name = config['run_name']
    if run_name == 'latest':
        run_name = sorted(os.listdir(config['package_path'] + f'logs/{config["exp_name"]}'))[-1]
    print(f'Run name: {run_name}')
    train_config = yaml.safe_load(open(config['package_path'] + f'logs/{config["exp_name"]}/{run_name}/train.yaml'))
    for key in train_config:
        if key not in config:
            config[key] = train_config[key]
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    test_loader, _, input_size, output_size = data_utils.get_dataloaders(config)
    model = getattr(models, config['model'])(input_size, output_size, config['hidden_dim'])
    model.load_state_dict(torch.load(config['package_path'] + f'logs/{config["exp_name"]}/{run_name}/model.pth'))
    acc_fn = globals()[f'get_accuracy_{config["acc_fn"]}']
    test_acc = test(model, test_loader, acc_fn)
    print(f'Test accuracy: {test_acc*100:.2f}%')

if __name__ == '__main__':
    run()