import torch
import torch.nn as nn
import numpy as np
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

import data_utils
import models
import os

def plot_loss(train_loss, val_loss, config):
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.title(f'{config["criterion"]} Loss, model: {config["model"]}, dataset: {config["dataset"]}')
    plt.savefig(config['package_path'] + f'logs/{config["exp_name"]}/{config["run_name"]}/loss.png')

def get_accuracy_CE(outputs, labels):
    return (outputs.argmax(dim=1) == labels).float().mean()

def get_accuracy_BCE(outputs, labels):
    return ((outputs > 0).float() == labels).float().mean()

def evaluate(model, val_loader, criterion, acc_fn, device, config):
    model.eval()
    model.to(device)
    running_loss = 0.0
    acc = 0
    with torch.no_grad():
        for i, data in enumerate(val_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            acc += acc_fn(outputs, labels)
    return running_loss / len(val_loader), acc / len(val_loader)

def train(model, train_loader, val_loader, optimizer, criterion, acc_fn, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    losses = {'train': [], 'val': []}
    for epoch in range(config['epochs']):
        running_loss = 0.0
        acc = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            acc += acc_fn(outputs, labels)
        losses['train'].append(running_loss / len(train_loader))
        if (epoch + 1) % config['val_freq'] == 0:
            val_loss, val_acc = evaluate(model, val_loader, criterion, acc_fn, device, config)
            losses['val'].append(val_loss)
            print(f'Epoch {epoch + 1}, Loss: {running_loss / len(train_loader)}, Val Loss: {val_loss}')
            print(f'Accuracy: {acc / len(train_loader)}, Val Accuracy: {val_acc}')
    plot_loss(losses['train'], losses['val'], config)
    torch.save(model.state_dict(), config['package_path'] + f'logs/{config["exp_name"]}/{config["run_name"]}/model.pth')
    np.save(config['package_path'] + f'logs/{config["exp_name"]}/{config["run_name"]}/train_losses.npy', losses['train'])
    np.save(config['package_path'] + f'logs/{config["exp_name"]}/{config["run_name"]}/val_losses.npy', losses['val'])
    return losses, acc, val_acc

def run(config):
    if config == None:
        config = yaml.safe_load(open('config/train.yaml'))
    config['run_name'] = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    print(f'Run name: {config["run_name"]}')
    os.makedirs(config['package_path'] + f'logs/{config["exp_name"]}/{config["run_name"]}', exist_ok=True)
    with open(config['package_path'] + f'logs/{config["exp_name"]}/{config["run_name"]}/train.yaml', 'w') as f:
        yaml.dump(config, f)

    torch.manual_seed(config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config['seed'])

    train_loader, val_loader, input_size, output_size = data_utils.get_dataloaders(config)
    
    model_class = getattr(models, config['model'])
    model = model_class(input_size, output_size, config['hidden_dim'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    criterion = getattr(nn, config['criterion'])()
    acc_fn = globals()[f'get_accuracy_{config["acc_fn"]}']

    losses, acc, val_acc = train(model, train_loader, val_loader, optimizer, criterion, acc_fn, config)
    return losses, acc, val_acc

if __name__ == '__main__':
    run(None)