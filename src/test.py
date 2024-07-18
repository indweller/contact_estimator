import torch
import torch.nn as nn
import numpy as np
import yaml
import os

import data_utils
import models
from common import *

def run(config):
    run_name = config['run_name']
    if run_name == 'latest':
        run_name = sorted(os.listdir(config['package_path'] + f'logs/{config["exp_name"]}'))[-1]
        config['run_name'] = run_name
    print(f'Testing on run name: {run_name}, exp name: {config["exp_name"]}')
    train_config = yaml.safe_load(open(config['package_path'] + f'logs/{config["exp_name"]}/{run_name}/train.yaml'))
    for key in train_config:
        if key not in config:
            config[key] = train_config[key]
    torch.manual_seed(config['seed'])
    np.random.seed(config['seed'])
    test_loader, _, input_size, output_size = data_utils.get_dataloaders(config)
    model = torch.load(os.path.join(config['package_path'], 'logs', config['exp_name'], run_name, 'model.pth'))
    if config["criterion"] == 'BCEWithLogitsLoss':
        pred_fn = get_prediction_BCE
    else:
        pred_fn = get_prediction_CE
    criterion = getattr(nn, config['criterion'])()
    test_loss, test_predictions, test_labels = evaluate(model, test_loader, criterion, pred_fn, config)
    test_acc = get_accuracy(test_predictions, test_labels)
    plot_confusion_matrix(test_predictions, test_labels, "test", config)
    print(f'Test accuracy: {test_acc*100:.2f}%')

if __name__ == '__main__':
    config = get_config('test')
    run(config)