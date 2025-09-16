import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import lightning as L
import mlflow
import mlflow.pytorch


class AdvancedMNISTCNN(L.LightningModule):
    def __init__(self, learning_rate=0.001, dropout=0.1):
        super().__init__()
        self.save_hyperparameters()
        
        # Stage 1: Initial feature extraction
        self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(8)
        self.trans1_1 = nn.Conv2d(8, 16, 1)
        self.bn1_1 = nn.BatchNorm2d(16)
        self.trans1_2 = nn.Conv2d(16, 16, 1)
        self.bn1_2 = nn.BatchNorm2d(16)

        # Stage 2: Mid-level features
        self.conv2 = nn.Conv2d(16, 20, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.trans2_1 = nn.Conv2d(20, 28, 1)
        self.bn2_1 = nn.BatchNorm2d(28)
        self.trans2_2 = nn.Conv2d(28, 28, 1)
        self.bn2_2 = nn.BatchNorm2d(28)

        # Stage 3: High-level features
        self.conv3 = nn.Conv2d(28, 36, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(36)
        self.trans3_1 = nn.Conv2d(36, 44, 1)
        self.bn3_1 = nn.BatchNorm2d(44)
        self.trans3_2 = nn.Conv2d(44, 32, 1)
        self.bn3_2 = nn.BatchNorm2d(32)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(32, 10)

    def forward(self, x):
        # Stage 1
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn1_1(self.trans1_1(x)))
        x = F.relu(self.bn1_2(self.trans1_2(x)))
        x = F.max_pool2d(x, 2)

        # Stage 2
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn2_1(self.trans2_1(x)))
        x = F.relu(self.bn2_2(self.trans2_2(x)))
        x = F.max_pool2d(x, 2)

        # Stage 3
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn3_1(self.trans3_1(x)))
        x = F.relu(self.bn3_2(self.trans3_2(x)))

        # GAP and prediction
        x = F.adaptive_avg_pool2d(x, 1)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', acc, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)


class MNISTDataModule(L.LightningDataModule):
    def __init__(self, batch_size=128, data_dir='../data', num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.num_workers = num_workers
        
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

    def setup(self, stage=None):
        full_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=True, download=True, transform=self.transform
        )
        test_dataset = torchvision.datasets.MNIST(
            root=self.data_dir, train=False, download=True, transform=self.transform
        )

        indices = torch.randperm(len(full_dataset))
        train_indices = indices[:50000]
        val_indices = indices[50000:]

        self.train_dataset = Subset(full_dataset, train_indices)
        self.val_dataset = Subset(full_dataset, val_indices)
        self.test_dataset = test_dataset

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, 
            batch_size=self.batch_size, shuffle=True, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=self.num_workers,
            persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)


def main():
    mlflow.set_tracking_uri("file:../experiments/mlruns")
    mlflow.set_experiment("mnist_experiments")
    mlflow.pytorch.autolog()
    
    config = {
        'learning_rate': 0.001,
        'dropout': 0.1,
        'batch_size': 128,
        'max_epochs': 20,
        'num_workers': 4,
        'data_dir': '../data'
    }
    
    model = AdvancedMNISTCNN(
        learning_rate=config['learning_rate'], 
        dropout=config['dropout']
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")
    print(f"Config: {config}")
    
    mlflow.log_params(config)
    mlflow.log_param("total_parameters", total_params)
    
    data_module = MNISTDataModule(
        batch_size=config['batch_size'],
        data_dir=config['data_dir'],
        num_workers=config['num_workers']
    )
    
    trainer = L.Trainer(
        max_epochs=config['max_epochs'],
        accelerator='auto',
        devices=1,
        deterministic=True
    )
    
    trainer.fit(model, data_module)
    
    val_results = trainer.validate(model, data_module)
    print(f"Final validation results: {val_results}")


if __name__ == "__main__":
    torch.manual_seed(42)
    main()
