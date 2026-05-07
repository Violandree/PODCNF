import numpy as np
import torch
from torch.utils.data import Dataset

def SinusoidalDensity(x, z):
  return x*torch.sin(3*np.pi*x)+2*z*x*(1-x)*(0.25-x)

def BimodalDensity(x, z, p):
  y1 = 10*x*(x-0.5)*(1.5-x) + z*0.3*(1.3-x)
  y2 = 10*x*(x-0.5)*(0.8-x) + z*0.3*(1.3-x)
  condition = (x < 0.5) | (p < 0.5)

  return torch.where(condition, y1, y2)

class SinusoidalData(Dataset):
    def __init__(self, num_points):
        self.num_points = num_points
        self._create_data()

    def __getitem__(self, item):
        return self.y[item]

    def __len__(self):
        return self.num_points

    def _create_data(self):
        self.x = torch.rand(self.num_points, 1)
        self.z = torch.randn(self.num_points, 1)

        self.y = SinusoidalDensity(self.x, self.z).to(torch.float32)

class BimodalData(Dataset):
    def __init__(self, num_points):
        self.num_points = num_points
        self._create_data()

    def __getitem__(self, item):
        return self.y[item]

    def __len__(self):
        return self.num_points

    def _create_data(self):
        self.x = torch.rand(self.num_points, 1)
        self.z = torch.randn(self.num_points, 1)
        self.p = torch.rand(self.num_points, 1)

        self.y = BimodalDensity(self.x, self.z, self.p).to(torch.float32)

class CombinedData(Dataset):
    def __init__(self, num_points):
        self.num_points = num_points
        self._create_data()

    def __getitem__(self, item):
        return self.data[item]

    def __len__(self):
        return self.num_points

    def _create_data(self):
        x = torch.rand(self.num_points, 1)

        z_sin = torch.randn(self.num_points, 1)

        z_bim = torch.randn(self.num_points, 1)
        p_bim = torch.rand(self.num_points, 1)

        y_sin = SinusoidalDensity(x, z_sin)
        y_bim = BimodalDensity(x, z_bim, p_bim)

        # first dim: x
        # second dim: y_sinusoidal
        # third dim: y_bimodal
        self.data = torch.cat([x, y_sin, y_bim], dim=1).to(torch.float32)