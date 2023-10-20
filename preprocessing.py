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



    def gfs_process(self):



    def era5_process(self):
