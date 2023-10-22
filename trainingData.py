import os
from datetime import datetime, timedelta

# Specify the directory where your GFS files are located
data_directory = "/path/to/your/gfs/files"

# Define the start and end datetime for the time range
start_datetime = datetime(2023, 1, 1, 0)  # Start date and time
end_datetime = datetime(2023, 1, 5, 18)   # End date and time

# Define the time step (every 6 hours)
time_step = timedelta(hours=6)

current_datetime = start_datetime

# Iterate over the time range and check for the presence of each file
while current_datetime <= end_datetime:
    date_str = current_datetime.strftime("%Y%m%d")
    hour_str = current_datetime.strftime("%H")
    file_name = f"GFS.t2m.{date_str}{hour_str}.nc"
    file_path = os.path.join(data_directory, file_name)

    if not os.path.exists(file_path):
        print(f"File missing: {file_name}")

    current_datetime += time_step
