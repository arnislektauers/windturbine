import rbf
import torch
import torch.nn as nn
import sys
import numpy as np
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from data_set import FeatureDataSet

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using {} device".format(device))


class Network(nn.Module):

    def __init__(self, layer_widths, layer_centres, basis_func):
        super(Network, self).__init__()
        self.rbf_layers = nn.ModuleList()
        self.linear_layers = nn.ModuleList()
        for i in range(len(layer_widths) - 1):
            self.rbf_layers.append(rbf.RBF(layer_widths[i], layer_centres[i], basis_func))
            self.linear_layers.append(nn.Linear(layer_centres[i], layer_widths[i + 1]))

    def forward(self, x):
        out = x
        for i in range(len(self.rbf_layers)):
            out = self.rbf_layers[i](out)
            out = self.linear_layers[i](out)
        return out

    def fit(self, trainset, epochs, batch_size, lr, loss_func):
        self.train()
        obs = len(trainset)
        trainloader = DataLoader(trainset, batch_size, shuffle=True)
        optimiser = torch.optim.Adam(self.parameters(), lr)
        epoch = 0
        while epoch < epochs:
            epoch += 1
            current_loss = 0
            batches = 0
            progress = 0
            for x_batch, y_batch in trainloader:
                batches += 1
                optimiser.zero_grad()
                y_hat = self.forward(x_batch)
                loss = loss_func(y_hat, y_batch)
                current_loss += (1 / batches) * (loss.item() - current_loss)
                loss.backward()
                optimiser.step()
                progress += y_batch.size(0)
                sys.stdout.write('\rEpoch: %d, Progress: %d/%d, Loss: %f      ' % (epoch, progress, obs, current_loss))
                sys.stdout.flush()


if __name__ == '__main__':
    # Generating a dataset for a given decision boundary

    samples = 400

    raining_set = FeatureDataSet(1, 400)
    test_set = FeatureDataSet(401, 444)

    # Instantiating and training an RBF network with the Gaussian basis function
    # This network receives a 2-dimensional input, transforms it into a 40-dimensional
    # hidden representation with an RBF layer and then transforms that into a
    # 1-dimensional output/prediction with a linear layer

    # To add more layers, change the layer_widths and layer_centres lists

    layer_widths = [3993, 1]
    layer_centres = [40]
    basis_func = rbf.gaussian

    rbf_net = Network(layer_widths, layer_centres, basis_func)
    rbf_net.fit(raining_set, epochs=300, batch_size=samples, lr=0.01, loss_func=nn.MSELoss())
    # BCEWithLogitsLoss
    rbf_net.eval()

    # Plotting the ideal and learned decision boundaries

  #  test_dataloader = DataLoader(test_set, batch_size=samples)
#
#    with torch.no_grad():
#        for X, y in test_dataloader:
#            xd, yd = X.to(device), y.to(device)
#            pred = rbf_net(xd)

#    fig, ax = plt.subplots(figsize=(16, 8), nrows=1, ncols=2)
#    plt.show()