import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L


class WideMNISTCNN(L.LightningModule):
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
        self.conv2 = nn.Conv2d(16, 20, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(20)
        self.trans2_1 = nn.Conv2d(20, 28, 1)
        self.bn2_1 = nn.BatchNorm2d(28)
        self.trans2_2 = nn.Conv2d(28, 28, 1)
        self.bn2_2 = nn.BatchNorm2d(28)

        # Stage 3: High-level features (ENHANCED - optimal width scaling)
        self.conv3 = nn.Conv2d(28, 41, 3, padding=1)  # 28→41 (+5 channels)
        self.bn3 = nn.BatchNorm2d(41)
        self.trans3_1 = nn.Conv2d(41, 49, 1)  # 41→49 (+5 channels)
        self.bn3_1 = nn.BatchNorm2d(49)
        self.trans3_2 = nn.Conv2d(49, 36, 1)  # 49→36 (+4 channels)
        self.bn3_2 = nn.BatchNorm2d(36)

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(36, 10)  # 36→10 (+4 input features)

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

        # Stage 3 (Enhanced)
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
        return torch.optim.Adam(
            self.parameters(), 
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )