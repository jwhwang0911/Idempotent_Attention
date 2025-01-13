from torch.utils.data import DataLoader
import torch
import numpy as np
from prefetch_generator import BackgroundGenerator


class DataLoaderX(DataLoader):
    def __iter__(self):
        return BackgroundGenerator(super().__iter__())

    def __getitem__(self, index):
        return self.dataset[index]

    def random_item(self):
        # indicies = [130, 400, 1024, 23]
        indicies = [1, 50, 100, 150]
        keys = self[0].keys()
        output = {}
        for key in keys:
            output[key] = torch.tensor(
                np.concatenate(
                    (
                        np.expand_dims(self.dataset[indicies[0]][key], axis=0),
                        np.expand_dims(self.dataset[indicies[1]][key], axis=0),
                        np.expand_dims(self.dataset[indicies[2]][key], axis=0),
                        np.expand_dims(self.dataset[indicies[3]][key], axis=0),
                    ),
                    axis=0,
                )
            )

        return output
