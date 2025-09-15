import marimo

__generated_with = "0.15.3"
app = marimo.App(width="full")


@app.cell
def _():
    import torch
    import torch.nn as nn
    return nn, torch


@app.cell
def _(nn):
    class TestNet(nn.Module):
      def __init__(self):
          super().__init__()
          self.fc = nn.Linear(10, 1)

      def forward(self, x):
          return self.fc(x)
    return (TestNet,)


@app.cell
def _(TestNet, nn, torch):
    model = TestNet()
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    return criterion, model, optimizer


@app.cell
def _(criterion, model, optimizer, torch):
    x = torch.randn(5, 10)
    y_target = x.mean(dim=1, keepdim=True)


    losses = []
    for epoch in range(20):
        y_pred = model(x)
        loss = criterion(y_pred, y_target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if epoch % 5 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    return (losses,)


@app.cell
def _(losses):
    print(f"Initial loss: {losses[0]:.4f}")
    print(f"Final loss: {losses[-1]:.4f}")
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
