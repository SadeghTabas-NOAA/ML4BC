import os
import boto3
import xarray as xr
import subprocess
import numpy as np
from datetime import datetime, timedelta, date, time
from botocore.config import Config
from botocore import UNSIGNED
import argparse
import fnmatch
import pygrib
import cdsapi

class DataProcessor:
    def __init__(self, start_date, end_date, output_directory=None, download_directory=None, keep_downloaded_data=False):
        self.start_date = start_date
        self.end_date = end_date
        self.output_directory = output_directory
        self.download_directory = download_directory
        self.keep_downloaded_data = keep_downloaded_data
        self.output_gfs = None
        self.output_era5 = None
        
        # Specify the output directory where you want to save the files
        if self.output_directory is None:
            self.output_directory= os.getcwd()
        
        # Initialize the S3 client
        self.s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))

        # Specify the S3 bucket name and root directory for GFS forecasts
        self.bucket_name = 'noaa-gfs-bdp-pds'
        self.root_directory = 'gfs'

        # Initialize the cdsapi client
        self.cds = cdsapi.Client()


    def gfs_process(self):
        # from datetime import time
        # Specify the local directory where you want to save the GFS files
        if self.download_directory is None:
            self.local_base_directory = os.path.join(os.getcwd(), 'noaa-gfs-bdp-pds-data')  # Use the current directory if not specified
        else:
            self.local_base_directory = os.path.join(self.download_directory, 'noaa-gfs-bdp-pds-data')
        os.makedirs(self.local_base_directory, exist_ok=True)
        
        self.output_gfs = os.path.join(self.output_directory, 'GFS')
        # Check if 'output_directory' exists and create it if it doesn't
        os.makedirs(self.output_gfs, exist_ok=True)
        
        # Loop through the 6-hour intervals
        current_datetime = datetime.combine(self.start_date, datetime.min.time())
        end_datetime = datetime.combine(self.end_date, time(18, 0, 0))
        while current_datetime <= end_datetime:
            date_str = current_datetime.strftime("%Y%m%d")
            time_str = current_datetime.strftime("%H")

            # Construct the S3 prefix for the directory
            s3_prefix = f"{self.root_directory}.{date_str}/{time_str}/{'atmos'}/{'gfs'}"

            # List objects in the S3 directory
            s3_objects = self.s3.list_objects_v2(Bucket=self.bucket_name, Prefix=s3_prefix)
            
            # Filter objects based on the desired formats
            mergedDSs = []
            for obj in s3_objects.get('Contents', []):
                obj_key = obj['Key']
                if fnmatch.fnmatch(obj_key, f'*.pgrb2.0p25.f00[0-2]'):
                    # Define the local directory path where the file will be saved
                    local_directory = os.path.join(self.local_base_directory, date_str, time_str)

                    # Create the local directory if it doesn't exist
                    os.makedirs(local_directory, exist_ok=True)

                    # Define the local file path
                    local_file_path = os.path.join(local_directory, os.path.basename(obj_key))

                    # Download the file from S3 to the local path
                    self.s3.download_file(self.bucket_name, obj_key, local_file_path)
                    print(f"Downloaded {obj_key} to {local_file_path}")

                    grbs = pygrib.open(local_file_path)
                    
                    # Specify the variable and level you want to extract
                    variable_name = '2 metre temperature'
        
                    # Find the matching grib message
                    variable_message = grbs.select(name=variable_name)[0]

                    # create a netcdf dataset using the matching grib message
                    lats, lons = variable_message.latlons()
                    data = variable_message.values
                    time = variable_message.validDate

                    ds = xr.Dataset(
                        data_vars={
                            't2m': (['lat', 'lon'], data)
                        },
                        coords={
                            'latitude': (['lat', 'lon'], lats),
                            'longitude': (['lat', 'lon'], lons),
                            'time': time,  
                        }
                    )
                    mergedDSs.append(ds)
                    
            # final_dataset = xr.merge(mergedDSs)
            final_dataset = xr.concat(mergedDSs, dim='time')
            
            # Define the output NetCDF file name
            output_file_name = f'GFS.t2m.{current_datetime.strftime("%Y%m%d%H")}.nc'
            output_file_path = os.path.join(self.output_gfs, output_file_name)
            final_dataset.to_netcdf(output_file_path)
            print(f"Saved the dataset to {output_file_path}")


    def era5_process(self):




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download and process GFS and ERA5 data")
    parser.add_argument("start_date", help="Start date in the format 'YYYYMMDD'")
    parser.add_argument("end_date", help="End date in the format 'YYYYMMDD'")
    parser.add_argument("-o", "--output", help="Output directory for processed data")
    parser.add_argument("-d", "--download", help="Download directory for raw data")
    parser.add_argument("-k", "--keep", help="Keep downloaded data (optional)", action="store_true", default=False)
    parser.add_argument("-p", "--process", nargs="*", choices=["gfs", "era5"], help="Specify which process to run")
    args = parser.parse_args()

    start_date = datetime.strptime(args.start_date, "%Y%m%d")
    end_date = datetime.strptime(args.end_date, "%Y%m%d")
    output_directory = args.output
    download_directory = args.download
    keep_downloaded_data = args.keep

    data_processor = DataProcessor(start_date, end_date, output_directory, download_directory, keep_downloaded_data)      
    if not args.process or "gfs" in args.process:
        data_processor.gfs_process()
    if not args.process or "era5" in args.process:
        data_processor.era5_process()
