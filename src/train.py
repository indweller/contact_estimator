import torch
import torch.nn as nn
import numpy as np
import yaml
from datetime import datetime
from tqdm import tqdm

import data_utils
import models
import os
from common import *

def train(model, train_loader, val_loader, optimizer, criterion, pred_fn, config):
    losses = {'train': [], 'val': []}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    with tqdm(range(config['epochs'])) as t:
        for epoch in t:
            model.train()
            train_loss, train_predictions, train_labels = train_one_epoch(model, train_loader, criterion, optimizer, pred_fn, config) 
            train_acc = get_accuracy(train_predictions, train_labels)       
            losses['train'].append(train_loss)
            if (epoch + 1) % config['val_freq'] == 0:
                val_loss, val_predictions, val_labels = evaluate(model, val_loader, criterion, pred_fn, config)
                val_acc = get_accuracy(val_predictions, val_labels)
                losses['val'].append(val_loss)
                tqdm.write(f'Epoch {epoch + 1}, Loss: {train_loss:.3f}, Val Loss: {val_loss:.3f}')
                tqdm.write(f'Accuracy: {train_acc * 100.0:.2f}%, Val Accuracy: {val_acc * 100.0:.2f}%')
            t.set_description(f"Epoch {epoch + 1}")
    plot_confusion_matrix(train_predictions, train_labels, "train", config)
    plot_confusion_matrix(val_predictions, val_labels, "val", config)
    plot_loss(losses['train'], losses['val'], config)
    torch.save(model, os.path.join(config['package_path'], 'logs', config['run_name'], 'model.pth'))
    torch.save(model, os.path.join(config['package_path'], 'logs', 'latest', 'model.pth'))
    np.save(config['package_path'] + f'logs/{config["run_name"]}/train_losses.npy', losses['train'])
    np.save(config['package_path'] + f'logs/{config["run_name"]}/val_losses.npy', losses['val'])
    return losses, train_acc, val_acc

def run(config):
    train_loader, val_loader, input_size = data_utils.get_dataloaders(config)
    model_class = getattr(models, config['model'])
    model = model_class(input_size, config['output_dim'], config['hidden_dim'])
    print(model)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = getattr(nn, config['criterion'])()
    if config["criterion"] == 'BCEWithLogitsLoss':
        pred_fn = get_prediction_BCE
    else:
        pred_fn = get_prediction_CE
    losses, train_acc, val_acc = train(model, train_loader, val_loader, optimizer, criterion, pred_fn, config)
    return losses, train_acc, val_acc

if __name__ == '__main__':
    config = get_config('train')
    torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config['seed'])
    run(config)