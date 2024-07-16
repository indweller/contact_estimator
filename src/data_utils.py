import torch
import torch.nn as nn
import numpy as np

def load_data(config, data_version=2):
    joint_state_data = np.load(config["package_path"] + f'data/processed/joint_state_data_{data_version}.npy')
    imu_data = np.load(config["package_path"] + f'data/processed/imu_data_{data_version}.npy')
    contact_data = np.load(config["package_path"] + f'data/processed/contact_data_{data_version}.npy')
    return joint_state_data, imu_data, contact_data

class SCMCDataset(torch.utils.data.Dataset):
    def __init__(self, joint_state_data, imu_data, contact_data):
        self.state = np.concatenate((joint_state_data, imu_data), axis=1)
        self.state = torch.tensor(self.state, dtype=torch.float32)
        self.contact = torch.tensor(contact_data, dtype=torch.long)
        self.contact = 2 * self.contact[:, 0] + self.contact[:, 1]

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        state = self.state[idx]
        contact = self.contact[idx]
        return state, contact
    
class SCMLDataset(torch.utils.data.Dataset):
    def __init__(self, joint_state_data, imu_data, contact_data):
        self.state = np.concatenate((joint_state_data, imu_data), axis=1)
        self.state = torch.tensor(self.state, dtype=torch.float32)
        self.contact = torch.tensor(contact_data, dtype=torch.float32)

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        state = self.state[idx]
        contact = self.contact[idx]
        return state, contact

def get_dataloaders(config):
    train_size = config['train_size']
    batch_size = config['batch_size']
    joint_state_data, imu_data, contact_data = load_data(config, config['data_version'])
    dataset_class = globals()[config['dataset'] + 'Dataset']
    if dataset_class == SCMLDataset:
        output_size = 2
    else:
        output_size = 4
    dataset = dataset_class(joint_state_data, imu_data, contact_data)
    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, dataset[0][0].shape[0], output_size