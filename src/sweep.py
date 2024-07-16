import matplotlib.pyplot as plt
import train
import yaml

config = yaml.safe_load(open('config/train.yaml'))
acc_fn = {'CrossEntropyLoss': 'CE', 'BCEWithLogitsLoss': 'BCE'}
datasets = {'CrossEntropyLoss': 'SCMC', 'BCEWithLogitsLoss': 'SCML'}
exp_losses = {} 
exp_accs = {}

def plot_losses(exp_losses):
    for exp_name, losses in exp_losses.items():
        plt.plot(losses['train'], label=f'{exp_name}_train', style='--')
        plt.plot(losses['val'], label=f'{exp_name}_val', style='-')
    plt.legend()
    plt.title('Losses')
    plt.tight_layout()
    plt.savefig('logs/sweep/losses.png')

def sweep(models, hidden_dims, loss_fns):
    for model in models:
        for hidden_dim in hidden_dims:
            for loss_fn in loss_fns:
                print(f'Training {model} with hidden_dim={hidden_dim} and loss_fn={loss_fn}')
                config['model'] = model
                config['hidden_dim'] = hidden_dim
                config['criterion'] = loss_fn
                config['dataset'] = datasets[loss_fn]
                config['acc_fn'] = acc_fn[loss_fn]
                config['exp_name'] = f'{model}_{hidden_dim}_{acc_fn[loss_fn]}'
                losses, _, val_acc = train.run(config)
                exp_losses[config['exp_name']] = losses
                exp_accs[config['exp_name']] = val_acc
    plot_losses(exp_losses)
    for exp_name, acc in exp_accs.items():
        print(f'{exp_name} accuracy: {acc*100:.2f}%')

if __name__ == '__main__':
    models = ['MLP']
    hidden_dims = [512, 128, 32]
    loss_fns = ['CrossEntropyLoss', 'BCEWithLogitsLoss']
    sweep(models, hidden_dims, loss_fns)