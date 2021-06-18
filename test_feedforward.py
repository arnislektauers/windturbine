import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_set import FeatureDataSet

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(3993, 50),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits


def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        #print(f"train {batch}")

        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"Training loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test(dataloader, test_model):
    size = len(dataloader.dataset)
    test_model.eval()
    test_loss, correct = 0, 0
    i = 0
    with torch.no_grad():
        for X, y in dataloader:
            xd, yd = X.to(device), y.to(device)
            pred = test_model(xd)
            test_loss += loss_fn(pred, yd).item()
            try:
                outputs = pred.argmax(dim=1, keepdim=True)
                correct += (outputs == yd).type(torch.float).sum().item()
            except RuntimeError as err:
                print("Correct check error: {0} i: {1}".format(err, i))
            i += 1
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")


if __name__ == '__main__':
    model = NeuralNetwork().to(device)
    print(model)

    loss_fn = nn.MSELoss()
    learning_rate = 1e-3
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    training_set = FeatureDataSet(1, 1000)
    test_set = FeatureDataSet(1001, 1101)

    batch_size = 10

    train_dataloader = DataLoader(training_set, batch_size=batch_size)
    test_dataloader = DataLoader(test_set, batch_size=batch_size)

    epochs = 20
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer)
        test(test_dataloader, model)
    print("Done!")
