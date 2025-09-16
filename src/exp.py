import argparse
import torch

from models.cnn import AdvancedMNISTCNN
# from models.rescnn import ResidualMNISTCNN
# from models.widecnn import WideMNISTCNN
from data.mnist import MNISTDataModule
from training.trainer import train_model


def main():
    parser = argparse.ArgumentParser(description='MNIST CNN Experiment')
    parser.add_argument('--desc', type=str, default="", help='Experiment description')
    parser.add_argument('--name', type=str, default="", help='Run name')
    args = parser.parse_args()
    
    config = {
        'learning_rate': 0.003,
        'dropout': 0.05,
        'batch_size': 32,
        'max_epochs': 20,
        'num_workers': 4,
        'data_dir': '../data',
        'weight_decay': 0,
    }
    
    model = AdvancedMNISTCNN(
        learning_rate=config['learning_rate'], 
        dropout=config['dropout'],
        weight_decay=config['weight_decay']
    )
    
    data_module = MNISTDataModule(
        batch_size=config['batch_size'],
        data_dir=config['data_dir'],
        num_workers=config['num_workers']
    )
    
    train_model(model, data_module, config, description=args.desc, run_name=args.name)


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
