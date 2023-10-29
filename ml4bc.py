import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from autoencoder_model import autoencoder
from netcdf_dataset import NetCDFDataset
from datetime import date

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("mps")

# Define your data directories
gfs_biased_dir = 'GFS/'
era5_unbiased_dir = 'ERA5/'

batch_size = 8
shuffle = False
num_workers = 0
seed = 42
torch.manual_seed(seed)

# Define the loss function and optimizer
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(autoencoder.parameters(), lr=0.0001)

# Create data loaders for GFS (biased) and ERA5 (unbiased) data
gfs_dataset = NetCDFDataset(root_dir=gfs_biased_dir)
era5_dataset = NetCDFDataset(root_dir=era5_unbiased_dir)

gfs_data_loader = DataLoader(gfs_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
era5_data_loader = DataLoader(era5_dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)


# Training loop with a custom progress bar
num_epochs = 50
for epoch in range(num_epochs):
    autoencoder.train()
    total_loss = 0.0

    # Create a custom progress bar for the epoch
    progress_bar = tqdm(enumerate(zip(gfs_data_loader, era5_data_loader)), total=len(gfs_data_loader), desc=f'Epoch [{epoch+1}/{num_epochs}]', dynamic_ncols=True)
    for batch_idx, (gfs_data, era5_data) in progress_bar:
        optimizer.zero_grad()
        outputs = autoencoder(gfs_data.to(device))
        loss = criterion(outputs, era5_data.to(device))
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    progress_bar.close()  # Close the custom progress bar

    # Calculate and print the average loss for the epoch
    avg_loss = total_loss / len(gfs_data_loader)
    print(f'Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}')

# Save the trained model
torch.save(autoencoder.module.state_dict() if isinstance(autoencoder, nn.DataParallel) else autoencoder.state_dict(), 'autoencoder_model.pth')
