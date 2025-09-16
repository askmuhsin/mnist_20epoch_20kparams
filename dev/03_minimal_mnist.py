import marimo

__generated_with = "0.15.3"
app = marimo.App(width="full")


@app.cell
def _():
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    import torchvision
    import torchvision.transforms as transforms
    from torch.utils.data import DataLoader, Subset
    import numpy as np
    return DataLoader, F, Subset, nn, torch, torchvision, transforms


@app.cell
def _(F, nn):
    class MinimalMNISTCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(1, 8, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(8)
            self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(16)
            self.conv3 = nn.Conv2d(16, 32, 3, padding=1)
            self.bn3 = nn.BatchNorm2d(32)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = F.relu(self.bn1(self.conv1(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn2(self.conv2(x)))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.bn3(self.conv3(x)))
            x = F.adaptive_avg_pool2d(x, 1)
            x = x.view(x.size(0), -1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    return (MinimalMNISTCNN,)


@app.cell
def _(MinimalMNISTCNN):
    model = MinimalMNISTCNN()
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params}")

    for name, param in model.named_parameters():
        print(f"{name}: {param.numel()} params")

    return (model,)


@app.cell
def _(DataLoader, Subset, torch, torchvision, transforms):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    full_dataset = torchvision.datasets.MNIST(
        root='../data', train=True, download=True, transform=transform
    )
    test_dataset = torchvision.datasets.MNIST(
        root='../data', train=False, download=True, transform=transform
    )

    indices = torch.randperm(len(full_dataset))
    train_indices = indices[:50000]
    val_indices = indices[50000:]

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    print(f"Train set: {len(train_dataset)} samples")
    print(f"Val set: {len(val_dataset)} samples") 
    print(f"Test set: {len(test_dataset)} samples")

    return train_loader, val_loader


@app.cell
def _():
    return


@app.cell
def _(model, nn, torch):
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    device = torch.device('mps' if torch.mps.is_available() else 'cpu')
    model.to(device)
    print(f"Using device: {device}")
    return criterion, device, optimizer


@app.cell
def _(torch):
    def train_epoch(model, train_loader, criterion, optimizer, device):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

        return total_loss / len(train_loader), 100. * correct / total

    def validate(model, val_loader, criterion, device):
        model.eval()
        total_loss = 0
        correct = 0
        total = 0

        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)

                total_loss += loss.item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                total += target.size(0)

        return total_loss / len(val_loader), 100. * correct / total

    return train_epoch, validate


@app.cell
def _(
    criterion,
    device,
    model,
    optimizer,
    train_epoch,
    train_loader,
    val_loader,
    validate,
):
    epochs = 20
    train_losses = []
    train_accs = []
    val_losses = []
    val_accs = []

    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)

        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print("-" * 50)

    return train_accs, val_accs


@app.cell
def _(train_accs, val_accs):
    print("Training Summary:")
    print(f"Final Train Accuracy: {train_accs[-1]:.2f}%")
    print(f"Final Val Accuracy: {val_accs[-1]:.2f}%")
    print(f"Best Val Accuracy: {max(val_accs):.2f}%")
    return


@app.cell
def _():


    return


if __name__ == "__main__":
    app.run()
