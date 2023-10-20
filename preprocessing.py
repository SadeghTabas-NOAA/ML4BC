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
