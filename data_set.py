import os
import pathlib
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from sklearn import preprocessing

APP_PATH = str(pathlib.Path(__file__).parent.resolve())


class FeatureDataSet(Dataset):

    def __init__(self, start_idx, end_idx):
        count = end_idx + 1 - start_idx
        x_train = np.zeros((count, 3993))
        y_train = np.zeros((count, 1))
        i = 0
        for idx in range(start_idx, end_idx + 1, 1):
            if idx % 100 == 0:
                print("Loading input files: {}".format(idx))

            matrix_file_name = os.path.join(APP_PATH, os.path.join("data3", "matrix_" + str(idx) + ".dat"))
            label_file_name = os.path.join(APP_PATH, os.path.join("data3", "label_" + str(idx) + ".dat"))

            matrix_file = pathlib.Path(matrix_file_name)
            label_file = pathlib.Path(label_file_name)

            if (not matrix_file.exists()) or (not label_file.exists()):
                if not matrix_file.exists():
                    print("Matrix file {} not found".format(matrix_file_name));
                if not label_file.exists():
                    print("Label file {} not found".format(label_file_name));
                continue

            matrix_data = pd.read_csv(matrix_file_name, header=None)
            label_data = pd.read_csv(label_file_name, header=None)

            # x_train = sc.fit_transform(matrix_data)

            x_train[i] = matrix_data.to_numpy().flatten()
            y_train[i] = label_data.to_numpy()[0].flatten()
            i += 1

        x_train = preprocessing.MaxAbsScaler().fit_transform(x_train)
        y_train = preprocessing.MinMaxScaler().fit_transform(y_train)

        self.x = torch.tensor(x_train, dtype=torch.float32)
        self.y = torch.tensor(y_train, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, index):
        x = self.x[index]
        y = self.y[index]
        return x, y


