import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class AdvancedMNISTCNN(L.LightningModule):
    def __init__(self, learning_rate=0.001, dropout=0.1, weight_decay=1e-4):
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
        self.conv2 = nn.Conv2d(16, 22, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(22)
        self.trans2_1 = nn.Conv2d(22, 30, 1)
        self.bn2_1 = nn.BatchNorm2d(30)
        self.trans2_2 = nn.Conv2d(30, 30, 1)
        self.bn2_2 = nn.BatchNorm2d(30)

        # Stage 3: High-level features
        self.conv3 = nn.Conv2d(30, 36, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(36)
        self.trans3_1 = nn.Conv2d(36, 46, 1)
        self.bn3_1 = nn.BatchNorm2d(46)
        self.trans3_2 = nn.Conv2d(46, 36, 1)
        self.bn3_2 = nn.BatchNorm2d(36)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(36, 10)

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
        optimizer = torch.optim.Adam(
            self.parameters(), 
            lr=0.003,  # Start with higher LR
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, 
            milestones=[5, 12, 18],  # 4-phase schedule  
            gamma=0.33               # 0.003→0.001→0.0003→0.0001
        )
        
        return [optimizer], [scheduler]
