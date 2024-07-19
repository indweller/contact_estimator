# List of common functions used in the project

import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

def get_config(file_name):
    package_path = os.path.dirname(os.path.abspath(__file__)) + '/'
    package_path = '/'.join(package_path.split('/')[:-2]) + '/'
    config = yaml.load(open(package_path + f'config/{file_name}.yaml', 'r'), Loader=yaml.FullLoader)
    config['package_path'] = package_path
    return config

def plot_loss(train_loss, val_loss, config):
    plt.plot(train_loss, label='train')
    plt.plot(val_loss, label='val')
    plt.legend()
    plt.title(f'{config["criterion"]} Loss, model: {config["model"]}, dataset: {config["dataset"]}')
    plt.tight_layout()
    plt.savefig(os.path.join(config['package_path'], 'logs', config['run_name'], 'loss.png'))
    plt.close()

def plot_confusion_matrix(predictions, labels, name, config):
    confusion_matrix = np.zeros((4, 4))
    predictions = predictions.int()
    labels = labels.int()
    if len(predictions.shape) == 1:
        for i in range(len(predictions)):
            confusion_matrix[predictions[i].item(), labels[i].item()] += 1
    else:
        for i in range(len(predictions)):
            confusion_matrix[(2 * predictions[i,0] + predictions[i,1]).item(), (2*labels[i, 0] + labels[i, 1]).item()] += 1
    fig, ax = plt.subplots()
    cax = ax.matshow(confusion_matrix)
    for i in range(4):
        for j in range(4):
            ax.text(i, j, str(confusion_matrix[i, j]), va='center', ha='center')
    plt.xlabel('True')
    plt.ylabel('Predicted')
    plt.xticks(range(4), ['0 0', '0 1', '1 0', '1 1'])
    plt.yticks(range(4), ['0 0', '0 1', '1 0', '1 1'])
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(config['package_path'], 'logs', config['run_name'], f'{name}_confusion_matrix.png'))
    plt.close()

def get_accuracy(predictions, labels):
    if len(predictions.shape) == 1:
        return (predictions == labels).float().mean().item()
    else:
        return (predictions == labels).all(dim=1).float().mean().item()

def get_prediction_CE(outputs):
    return outputs.argmax(dim=1)

def get_prediction_BCE(outputs):
    return (outputs > 0).float()

def evaluate(model, loader, criterion, pred_fn, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.eval()
    model.to(device)
    running_loss = 0.0
    all_predictions, all_labels = [], []
    with torch.no_grad():
        for i, data in enumerate(loader, 0):
            inputs, labels = data
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            predictions = pred_fn(outputs)
            all_predictions.append(predictions)
            all_labels.append(labels)
    return running_loss / len(loader), torch.cat(all_predictions, dim=0), torch.cat(all_labels, dim=0)

def train_one_epoch(model, loader, criterion, optimizer, pred_fn, config):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train()
    model.to(device)
    running_loss = 0.0
    all_predictions, all_labels = [], []
    for i, data in enumerate(loader, 0):
        inputs, labels = data
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        predictions = pred_fn(outputs)
        all_predictions.append(predictions)
        all_labels.append(labels)
    return running_loss / len(loader), torch.cat(all_predictions, dim=0), torch.cat(all_labels, dim=0)