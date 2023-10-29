import netCDF4 as nc
import numpy as np
import os
from torch.utils.data import Dataset
from datetime import timedelta, date

class NetCDFDataset(Dataset):
    def __init__(self, root_dir, start_date, end_date, transform=True):
        self.root_dir = root_dir
        self.file_list = self.create_file_list(root_dir, start_date, end_date)
        self.mean = 278.83097 #279.9124
        self.std = 56.02780 #107.1107
        self.transform = transform

    @staticmethod
    def create_file_list(root_dir, start_date, end_date):
        file_list = []
        time_step = timedelta(days=1)
        current_date = start_date

        while current_date <= end_date:
            for hh in ['00', '06', '12', '18']:
                filename = f'{os.path.basename(root_dir)}.t2m.{current_date.strftime("%Y%m%d")}{hh}.nc'
                file_list.append(filename)
            current_date += time_step
        
        return file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = os.path.join(self.root_dir, self.file_list[idx])
        
        # Load NetCDF data
        dataset = nc.Dataset(file_path)
        data = dataset.variables['t2m'][:].astype(np.float32)  # Adjust 'data' to the variable name in your file
        dataset.close()
        
        # Reshape the data to (1, 50, 721, 1440)
        data = data.reshape(1, 50, 721, 1440)

        if self.transform:
            data = self.normalize_data(data)  # Normalize the data if transform is True

        return torch.tensor(data)

    def normalize_data(self, data):
        data = (data - self.mean) / self.std
        return data

    def rescale_data(self, data):
        data = (data * self.std) + self.mean
        return data
