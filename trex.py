from dataclasses import dataclass, astuple, asdict,field
from importlib.metadata import metadata
import numpy as np
import math
import datetime as dt
import pathlib
import re
import h5py
from scipy import signal, interpolate
from obspy.io.segy.core import _read_segy
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import warnings
import sys

def search_for_file(shots_directory, timestamp):
    shot_files = []
    for root, dirs, files in os.walk(shots_directory):
        for file in files:
            if(file.endswith(".sgy")):
                file_path = os.path.join(root,file)
                shot_file = pathlib.Path(file_path)
                shot_files.append(shot_file)
                
    shot_files = sorted(shot_files)
    file_names_and_start_times = {}
    for shot_file in shot_files:
        shot_file = pathlib.Path(shot_file)
        match =re.search(r'\d{6}_\d{6}', shot_file.stem)
        start_time = dt.datetime.strptime(match.group(), '%y%m%d_%H%M%S')
        file_names_and_start_times[shot_file] = start_time

    for file_name, file_start_time in file_names_and_start_times.items():
        if file_start_time == timestamp:
            print(f"Trex shot file is {file_name}")
            break
    return(file_name)

def trim(file_path, event_duration):
    shot = _read_segy(file_path)
    start_time = shot[0].stats.starttime
    end_time = start_time + event_duration - shot[0].stats.delta
    shot.trim(start_time,end_time)
    return(shot)


def convert_to_vaults(file_path):
    shot = _read_segy(file_path)

    for n, channel  in enumerate(shot):
        if n == 0:
            factor = 12
        elif n == 1:
            factor = 156.6
        elif n == 2:
            factor = 317.2
        channel.data = channel.data / factor
    return (shot)



def convert_shot_to_vaults(shot):
    for n, channel  in enumerate(shot):
        if n == 0:
            factor = 12
        elif n == 1:
            factor = 156.6
        elif n == 2:
            factor = 317.2
        channel.data = channel.data / factor
    return (shot)



    # convert to engineering units
def convert_to_engineering_units_english(file_path, direction, trim = False, event_duration = None):
    if trim:
        shot = trim(file_path, event_duration)
    else:
        shot = _read_segy(file_path)

    shot = convert_shot_to_vaults(shot)

    if direction == "Z":
        base_plate_weight = 4100 
        moving_mass_weight = 8100 
    elif direction == "X" or direction == "Y":
        base_plate_weight = 7400  
        moving_mass_weight = 4850

    shot[1].data = shot[1].data / 0.2  # convert to acceleration in g's
    shot[2].data = shot[2].data / 0.2  # convert to acceleration in g's
    shot[3].data = shot[1].data* base_plate_weight + shot[2].data * moving_mass_weight # convert to Force in lb
    return(shot)






def convert_to_engineering_units_SI(file_path, direction, trim_channels = False, event_duration = None):
    if trim_channels:
        shot = trim(file_path, event_duration)
    else:
        shot = _read_segy(file_path)

    shot = convert_shot_to_vaults(shot)

    if direction == "Z":
        base_plate_mass = 1860  
        moving_mass_mass = 3670  
    elif direction == "X" or direction == "Y":
        base_plate_mass = 3357   
        moving_mass_mass = 2200 

    shot[1].data = shot[1].data / 0.2  # convert to acceleration in g's  
    shot[2].data = shot[2].data / 0.2  # convert to acceleration in g's
    shot[3].data = ((shot[1].data * base_plate_mass * 9.81)/1000) + ((shot[2].data* moving_mass_mass * 9.81)/1000) # convert to Force in kN
    return(shot)


