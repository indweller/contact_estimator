import torch
import numpy as np

def load_data(config, data_version=2):
    state_data = np.load(config["package_path"] + f'data/processed/state_data_{data_version}.npy')
    contact_data = np.load(config["package_path"] + f'data/processed/contact_data_{data_version}.npy')
    return state_data, contact_data

class ContactDataset(torch.utils.data.Dataset):
    def __init__(self, state_data, contact_data, output_dim=2):
        self.state = torch.tensor(state_data, dtype=torch.float32)
        if output_dim == 2:
            self.contact = torch.tensor(contact_data, dtype=torch.float32)
        else:
            self.contact = torch.tensor(contact_data, dtype=torch.long)
            self.contact = 2 * self.contact[:, 0] + self.contact[:, 1]

    def __len__(self):
        return len(self.state)

    def __getitem__(self, idx):
        state = self.state[idx]
        contact = self.contact[idx]
        return state, contact
    
def get_dataloaders(config):
    train_size = config['train_size']
    batch_size = config['batch_size']
    state_data, contact_data = load_data(config, config['data_version'])
    dataset = ContactDataset(state_data, contact_data, config['output_dim'])
    train_size = int(train_size * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=12, pin_memory=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=12, pin_memory=True)
    print(f"Input size: {dataset[0][0].shape[0]}")
    return train_loader, val_loader, dataset[0][0].shape[0]