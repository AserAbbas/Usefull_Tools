from dataclasses import dataclass, astuple, asdict,field
import numpy as np
import math
import datetime as dt
import pathlib
import re
import h5py
from scipy import signal, interpolate
import matplotlib.pyplot as plt
import pandas as pd
import datetime
import os
import warnings
import sys


def fetch_parameters_from_h5file(file_path):
    assert file_path.exists(),"The provided file path doesn't exist"
    with h5py.File(file_path,"r") as f:
        gauge_length = f['Acquisition'].attrs['GaugeLength']
        fibre_refractive_index = f['Acquisition/Custom'].attrs['Fibre Refractive Index']
        laser_wavelength = f['Acquisition/Custom'].attrs['Laser Wavelength (nm)']
        number_of_channels = f['Acquisition/Custom'].attrs['Num Output Channels']
        rawdata_sampling_rate = int(f['Acquisition/Raw[0]'].attrs['OutputDataRate'])
        file_duration = f['Acquisition/Raw[0]/RawDataTime'].attrs['Count']/rawdata_sampling_rate
        spatial_sampling_interval =  f['Acquisition'].attrs['SpatialSamplingInterval']
    return(gauge_length,fibre_refractive_index,laser_wavelength,number_of_channels,rawdata_sampling_rate,file_duration,spatial_sampling_interval)





#-------------------------------------------------------------------------
def inspect_h5_content(file_path):
    """returns the h5 file group names, datasets, and their attributes

    Args:
        file_path (str): path to the file to be inspected
    """
    # assert file_path.exists(),"The provided file path doesn't exist"
    with h5py.File(file_path,"r") as f:
        print("Following are the files inside this h5 file: ")
        print("_______________________________________________")

        f.visit(_printname) #iterates over all the files in the hdf5 file
        print("\nFiles content: ")
        print("_______________________________________________")
        f.visititems(_pr_values)
        
        print("\nGroup attributes: ")
        print("_______________________________________________")
        f.visititems(_attributes)

def _printname(name):
    """helper function to the inspect_h5_content
    """
    print(name)


def _pr_values(name,f):
    """helper function to the inspect_h5_content
    """
    # prints the values of the files in the main array
    if not isinstance(f,h5py.Group):
        print(f'\n{name}') #prints the name of the array
        data = f[:] #extracts the data from the file and store it in an array with the name "data"
        print(data.shape)
        print(data.dtype)
        
def _attributes(name,f):
    """helper function to the inspect_h5_content
    """
    print(f'\n{name}\n__________________') #prints the name of the array
    group = f
    for key in group.attrs.keys():
        print('{} : {}'.format(key, group.attrs[key]))
#--------------------------------------------------------------------------        
        
        
@dataclass
class Shot:
    name: str 
    timestamp: dt.datetime 
    mode: str = None
    direction: str = None
    sampling_rate: int = None
    lat: str = None
    lon: str = None
    local_x: str = None
    local_y: str = None
    source: str = None
    config: str = None
    ch1: np.ndarray = None
    ch1_description: str = None
    ch2: np.ndarray = None
    ch2_description: str = None
    ch3: np.ndarray = None
    ch3_description: str = None
    ch4: np.ndarray = None
    ch4_description: str = None


class Event:
    """An Event combines a Shot object and its associated wavefield collected by the DAS caple.
    """
    def __init__(self, shot,  event_duration,no_of_channels,sampling_datetime,data, sampling_rate, resampled_from = None, data_type= 'raw data',spatial_sampling_interval =1.0209523439407349, gauge_length =  2.0419046878814697,lowpass_filtered = False,lowpass_frequency = None,highpass_filtered=False,highpass_frequency=None,downsampled= False, datetime_description = None):
        """Creates an event from a Shot object and wavefield information

        Args:
            shot (object): can be created using the Shot class
            event_duration (float): the waveforms will be trimmed to be this value
            no_of_channels (int): number of channels resembled by the DAS cable
            sampling_datetime (1D array): array of the samples date and time
            datetime_description (str, optional): datetime data type. 
            sampling_rate (int): the data sampling rate
            data_type (str, optional): the wavefield type/units. Defaults to 'raw data'
            spatial_sampling_interval (float, optional): the spatial sampling interval (channel separation). Defaults to 1.0209523439407349.
            gauge_length (float, optional): the guage length. Defaults to 2.0419046878814697.
            lowpass_filtered (bool, optional): whether or not the wavefield has been lowpass filtered. Defaults to False.
            lowpass_frequency (_type_, optional): if it has been lowpass filtered, which frequency was used. Defaults to None.
            highpass_filtered (bool, optional): whether or not the wavefield has been Highass filtered. Defaults to False.
            highpass_frequency (_type_, optional): if it has been highpass filtered, which frequency was used. Defaults to None.
            downsampled (bool, optional): whether or not the wavefield has been downsampled. Defaults to False.
        """
        
        
        shot: Shot = shot
        event_duration: float = event_duration
        no_of_channels: int = no_of_channels  
        sampling_datetime:  np.ndarray =  sampling_datetime
        datetime_description: str = datetime_description
        data: np.ndarray = data
        resampled_from: int = resampled_from
        sampling_rate: int = sampling_rate
        data_type: str = data_type
        spatial_sampling_interval: float  = spatial_sampling_interval
        gauge_length: float = gauge_length
        lowpass_filtered: bool = lowpass_filtered
        lowpass_frequency: float = lowpass_frequency
        highpass_filtered: bool = highpass_filtered
        highpass_frequency: float = highpass_frequency
        downsampled: bool = downsampled
 

    def __repr__(self) -> str:
        return (f"{self.shot}, event_duration = {self.event_duration}, no_of_channels = {self.no_of_channels}, sampling_rate = {self.sampling_rate}, data_type = {self.data_type}, datetime_description = {self.datetime_description}, spatial_sampling_interval = {self.spatial_sampling_interval}, gauge_length = {self.gauge_length}, lowpass_filtered = {self.lowpass_filtered}, lowpass_frequency = {self.lowpass_frequency} ,highpass_filtered = {self.highpass_filtered}, highpass_frequency = {self.highpass_frequency} ,downsampled = {self.downsampled}, resampled_from = {self.resampled_from}")


    def search_files_for_event(self,waveforms_directory,file_duration = 60, pretrigger_delay = 0):
        """Determines the H5 files containing a given shot based on the shot start and end time. 

        Args:
            waveforms_directory (strin): The path of the directory containing the H5 files
            pretrigger_delay (float, optional): The event start time will be the shot start time minus the pre-trigger delay. Defaults to 0.0.
        """
        
        
        assert pathlib.Path(waveforms_directory).exists(),"The provided directory doesn't exist"
        
        # read H5 files within the provided directory and creata a dictionary with times and dates
        waveform_files = []
        for root, dirs, files in os.walk(waveforms_directory):
            for file in files:
                if(file.endswith(".h5")):
                    file_path = os.path.join(root,file)
                    waveform_file = pathlib.Path(file_path)
                    waveform_files.append(waveform_file)
        
        waveform_files = sorted(waveform_files)
        file_names_and_start_times = {}
        for waveform_file in waveform_files:
            waveform_file = pathlib.Path(waveform_file)
            match =re.search(r'\d{8}_\d{6}.\d{3}', waveform_file.stem)
            start_time = dt.datetime.strptime(match.group(), '%Y%m%d_%H%M%S.%f')
            file_names_and_start_times[waveform_file] = start_time

            
        # determine event start and endtime using the shot timestamp and event duration 
        trim_start_time = self.shot.timestamp - dt.timedelta(0,pretrigger_delay)
        trim_end_time = trim_start_time+ dt.timedelta(0,self.event_duration)
        file_duration = file_duration
        # check if the shot exists within the defined folder
        if list(file_names_and_start_times.values())[len(file_names_and_start_times)-1] < trim_start_time or list(file_names_and_start_times.values())[0] > trim_start_time:
            print( "This shot is not in this folder")
            event_file_names_and_start_times=[]
        else:
            file_names=[]
            event_file_names_and_start_times={}
            for file_name, file_start_time in file_names_and_start_times.items():
                if file_start_time < trim_start_time and file_start_time+dt.timedelta(0,file_duration) > trim_end_time:
                    file_names.append(file_name)
                    event_file_names_and_start_times[file_name] = file_start_time
                    break
                if file_start_time < trim_start_time and file_start_time+dt.timedelta(0,file_duration+ self.event_duration) > trim_end_time:
                    file_names.append(file_name)
                    event_file_names_and_start_times[file_name] = file_start_time
                if file_start_time > trim_start_time and file_start_time < trim_end_time:
                    file_names.append(file_name)
                    event_file_names_and_start_times[file_name] = file_start_time
                    break
            gauge_length,fibre_refractive_index,laser_wavelength,number_of_channels,rawdata_sampling_rate,file_duration_check,spatial_sampling_interval = fetch_parameters_from_h5file(file_names[0])
            if file_duration_check >1.1*file_duration or file_duration_check < .9*file_duration:
                sys.exit("The specified file duration doesnt match the actual file duration")

                
            self.sampling_rate =  int(rawdata_sampling_rate)
            self.no_of_channels = number_of_channels
            print(f'Shot {self.shot.name} direction {self.shot.direction} time {trim_start_time}')
            print(f'Files containing the shot {file_names}')
            return(event_file_names_and_start_times)
        
    
    
    
    
    
    
    def lowpass_filter(self,lowpass_frequency,filter_order = 40):
        """Lowpass filter the wavefield using a bi-directional SOS ButterWorth filter  

        Args:
            lowpass_frequency (float): remove frequencies above this value
            filter_order (int, optional): the order of the ButterWorth filter. Defaults to 40.
        """
        self.lowpass_filtered = True
        self.lowpass_frequency = lowpass_frequency
        w_c = 2*lowpass_frequency/self.sampling_rate
        lowpass_ButterWorth_filter = signal.butter(N = filter_order, Wn =  w_c, btype='low', analog=False, output='sos')
        for i in range(self.data.shape[0]):
            self.data[i,:] -= self.data[i,0]
            self.data[i,:] = signal.sosfiltfilt(lowpass_ButterWorth_filter,self.data[i,:])
        
        if(np.isnan(self.data).any()):
            print("The wavefield contain NaN values")
        return(self)
       
    
    
    
    
    
    
    def highpass_filter(self,highpass_frequency,filter_order = 80):
        """Highpass filter the wavefield using a bi-directional SOS ButterWorth filter

        Args:
            highpass_frequency (float): remove frequencies below this value
            filter_order (int, optional): the order of the ButterWorth filter. Defaults to 80.
        """
        self.highpass_filtered = True
        self.highpass_frequency = highpass_frequency
        w_c = 2*highpass_frequency/self.sampling_rate
        highpass_ButterWorth_filter = signal.butter(N = filter_order, Wn =  w_c, btype='highpass', analog=False, output='sos')
        for i in range(self.data.shape[0]):
            self.data[i,:] = signal.sosfiltfilt(highpass_ButterWorth_filter,self.data[i,:])
        
        if(np.isnan(self.data).any()):
            print("The wavefield contain NaN values")
        return(self)
            
    
    
    
    def downsample(self,new_sampling_rate):
        """Downsample the wavefield

        Args:
            new_sampling_rate (int): downsample the wavefiled to this new sampling rate
        """
        assert new_sampling_rate<=self.sampling_rate, "Can't downsample because the new sampling rate is higher than the existing one"
        step = int(self.sampling_rate/new_sampling_rate)
        self.data = self.data[:,0:self.data.shape[1]:step]
        self.sampling_datetime = self.sampling_datetime[0:self.sampling_datetime.shape[0]:step]
        self.downsampled = True
        self.resampled_from = self.sampling_rate
        self.sampling_rate = new_sampling_rate
        return(self)
    
    
    
    
    
    
    
    def strain_from_raw_data(self,laser_wavelength,fibre_refractive_index,gauge_length,rawdata_to_phase_factor, Photoelastic_Scaling_Factor = 0.78 ):
        """Returns the microstrain using the rawdata and the following parameters

        Args:
            laser_wavelength (float): can be found in the H5 header info
            fibre_refractive_index (float): can be found in the H5 header info
            gauge_length (float): can be found in the H5 header info
            Photoelastic_Scaling_Factor (float, optional): Defaults to 0.78
            rawdata_to_phase_factor (float): can be found in the H5 header info
        """

        if self.data_type == "raw data":         
            self.data= ((rawdata_to_phase_factor*laser_wavelength*self.data)/(4*np.pi*fibre_refractive_index*gauge_length*10**9*Photoelastic_Scaling_Factor))*10**6

        elif self.data_type == "phase(rad)":
            self.data= (self.data*laser_wavelength*10**6)/(4*math.pi*fibre_refractive_index*gauge_length*10**9*Photoelastic_Scaling_Factor)
        self.data_type = "micro strain"    
        return(self)
          
        
    
    def phase_from_raw_data(self,rawdata_to_phase_factor):
        """Returns the phase from the rawdata

        Args:
            rawdata_to_phase_factor (float): can be found in the H5 header info
        """
        if self.data_type == "raw data":
            self.data= rawdata_to_phase_factor*self.data
            self.data_type = "phase(rad)"
        else:
            print(f"The data has not been converted to phase because it is in {self.data_type} and not in its raw form.")
            
        return(self)
    
    
    
    def phase_from_strain(self,laser_wavelength,fibre_refractive_index,gauge_length,Photoelastic_Scaling_Factor):
        """Returns the phase from the microstrain wavefield

        Args:
            laser_wavelength (_type_): can be found in the H5 header info
            fibre_refractive_index (_type_): can be found in the H5 header info
            gauge_length (_type_): can be found in the H5 header info
            Photoelastic_Scaling_Factor (_type_): Defaults to 0.78
        """
        if self.data_type == "micro strain":
            self.data = ((self.data/10**6) *4*np.pi*fibre_refractive_index*gauge_length*10**9*Photoelastic_Scaling_Factor)/laser_wavelength
            self.data_type = "phase(rad)"
        else:
            print(f"The data has not been converted to phase because it is in {self.data_type} and not in strain.")
        return(self)
    
    
    @classmethod
    def from_h5dir(cls, shot , event_duration, waveforms_directory , file_duration = 60, pretrigger_delay = 0.0, lowpass_filter = False,
                     lowpass_frequency = None,lowpass_filter_order = 40,highpass_filter = False,
                     highpass_frequency = None, highpass_filter_order = 80, downsample = False, resampled_from = None,
                     new_sampling_rate = None, convert_to_strain = False, convert_to_phase =False,
                     Photoelastic_Scaling_Factor=0.78, rawdata_to_phase_factor = None, outer_trim = True, datetime_description = "Unix format"):
        """Creates an event using a shot object and the name of the directory that contains the H5 files

        Args:
            shot (object): can be created using the Shot class
            event_duration (float): the waveforms will be trimmed to be this value
            waveforms_directory (string): the path to the directory hosting the H5 files
            pretrigger_delay (float, optional): The event start time will be the shot start time minus the pre-trigger delay. Defaults to 0.0
            lowpass_filter (bool, optional): Lowpass filter the wavefield using a bi-directional SOS ButterWorth filter. Defaults to False.
            lowpass_frequency (float, optional): remove frequencies above this value. Defaults to None.
            lowpass_filter_order (int, optional): the order of the ButterWorth filter. Defaults to 40.
            highpass_filter (bool, optional): Highpass filter the wavefield using a bi-directional SOS ButterWorth filter. Defaults to False.
            highpass_frequency (_type_, optional): remove frequencies below this value. Defaults to None.
            highpass_filter_order (int, optional): the order of the ButterWorth filter. Defaults to 80.
            downsample (bool, optional): Downsample the wavefield. Defaults to False.
            new_sampling_rate (int, optional): downsample the wavefield to this rate. Defaults to None.
            convert_to_strain (bool, optional): converts the wavefield to a microstrain wavefield. Defaults to False.
            convert_to_phase (bool, optional): converts the wavefield to phase wavefiled. Defaults to False.
            Photoelastic_Scaling_Factor (float, optional): Defaults to 0.78.
            rawdata_to_phase_factor (_type_, optional): can be found in the H5 header info. Defaults to None.
            outer_trim (bool, optional): Trim the time series and include the entire signal or use the closest data points to the start and end times
        """

        #initializing the default values
        cls.shot = shot
        cls.event_duration= event_duration
        cls.no_of_channels = None       
        cls.data_type ='raw data'
        cls.lowpass_filtered = False
        cls.lowpass_frequency = None
        cls.highpass_filtered = False
        cls.highpass_frequency = None
        cls.downsampled= False
        cls.sampling_rate = None
        cls.resampled_from = None
        cls.datetime_description = datetime_description
            
        
        file_names_and_start_times= cls.search_files_for_event(cls, waveforms_directory,file_duration, pretrigger_delay)
        if len(file_names_and_start_times) == 0:
            return(cls)
        
        else:
            # determine event start and endtime using the shot timestamp and event duration 
            trim_start_time = shot.timestamp - dt.timedelta(0,pretrigger_delay)
            trim_end_time = trim_start_time+ dt.timedelta(0,cls.event_duration)
            
            gauge_length,fibre_refractive_index,laser_wavelength,number_of_channels,rawdata_sampling_rate,file_duration,spatial_sampling_interval = fetch_parameters_from_h5file(list(file_names_and_start_times.keys())[0])
            cls.sampling_rate = rawdata_sampling_rate
            cls.gauge_length = gauge_length
            cls.spatial_sampling_interval = spatial_sampling_interval


            
            if downsample:
                assert new_sampling_rate <= cls.sampling_rate, "Can't downsample because the new sampling rate is higher than the existing one"


            # open located files and trim out the shot record
                # create empty array to house the data
            record_length = int(cls.event_duration * rawdata_sampling_rate)
            if outer_trim:
                cls.sampling_datetime = np.empty((record_length+1), dtype = np.int64)
                cls.data = np.empty((number_of_channels, record_length+1), dtype = np.int32)
            else:
                cls.sampling_datetime = np.empty((record_length), dtype = np.int64)
                cls.data = np.empty((number_of_channels, record_length), dtype = np.int32)
                       
            for count, file_name in enumerate(file_names_and_start_times):
                with h5py.File(file_name,"r") as f:

                    assert  f['Acquisition'].attrs['GaugeLength'] == gauge_length, "The files in the list have varying guage length"
                    assert  f['Acquisition/Custom'].attrs['Fibre Refractive Index'] == fibre_refractive_index, "files have varying Fibre Refractive Index"
                    assert  f['Acquisition/Custom'].attrs['Laser Wavelength (nm)'] == laser_wavelength, "files have varying laser_wavelength"
                    assert  f['Acquisition/Custom'].attrs['Num Output Channels'] == number_of_channels, "files have varying number of channels"
                    assert  f['Acquisition/Raw[0]'].attrs['OutputDataRate'] == rawdata_sampling_rate, "files have varying sampling rate"
                    # assert  (f['Acquisition/Raw[0]/RawDataTime'].attrs['Count']/rawdata_sampling_rate) == file_duration, "files have varying duration"
                    #assert  f['Acquisition/Raw[0]'].attrs['RawDataUnit'] == b'rad * 2PI/2^16'
                        
                    
                    
                    if count == 0:
                        file_start_time =datetime.datetime.utcfromtimestamp(f["Acquisition/Raw[0]/RawDataTime"][0]/1000000)
                        if outer_trim:
                            start_index = math.floor(((trim_start_time-file_start_time).total_seconds())*rawdata_sampling_rate)
                            first_file_end_index = math.ceil(((trim_end_time-file_start_time).total_seconds())*rawdata_sampling_rate)
                        else:
                            start_index = round(((trim_start_time-file_start_time).total_seconds())*rawdata_sampling_rate)
                            first_file_end_index = round(((trim_end_time-file_start_time).total_seconds())*rawdata_sampling_rate)
                        if first_file_end_index > (file_duration*rawdata_sampling_rate):
                            first_file_end_index = int(file_duration*rawdata_sampling_rate)   
                        cls.sampling_datetime[0:(first_file_end_index-start_index)] = f["Acquisition/Raw[0]/RawDataTime"][start_index:first_file_end_index]
                        cls.data[:,0:(first_file_end_index-start_index)] = f["Acquisition/Raw[0]/RawData"][:,start_index:first_file_end_index]

                    
                    if count == 1:  
                        file_start_time =datetime.datetime.utcfromtimestamp(f["Acquisition/Raw[0]/RawDataTime"][0]/1000000)


                        if outer_trim:
                            # end_index = math.ceil(((trim_end_time-file_start_time).total_seconds())*rawdata_sampling_rate)
                            end_index = record_length - (first_file_end_index-start_index) +1
                            cls.sampling_datetime[(first_file_end_index-start_index):record_length+1] = f["Acquisition/Raw[0]/RawDataTime"][0:end_index]
                            cls.data[:,(first_file_end_index-start_index):record_length+1] = f["Acquisition/Raw[0]/RawData"][:,0:end_index]
                        else:
                            # end_index = round(((trim_end_time-file_start_time).total_seconds())*rawdata_sampling_rate)
                            end_index = record_length - (first_file_end_index-start_index)
                            cls.sampling_datetime[(first_file_end_index-start_index):record_length] = f["Acquisition/Raw[0]/RawDataTime"][0:end_index]
                            cls.data[:,(first_file_end_index-start_index):record_length] = f["Acquisition/Raw[0]/RawData"][:,0:end_index]
            
            
            
            f_start_t = cls.sampling_datetime[0]
            check_shot_duration = cls.sampling_datetime.shape[0]-1
            f_end_t = cls.sampling_datetime[check_shot_duration]
            file_timespan = f_end_t-f_start_t
            shot_timespan = f_end_t-f_start_t

            # code to check if the file has missing samples
            # TODO: Aser figure out how to programaticall calculate the factor instead of this way
            if rawdata_sampling_rate == 10000:
                factor = 100
            else:
                factor =1000

            if outer_trim:
                outer_trim_factor = 0
            else:
                outer_trim_factor = 1

            if shot_timespan != (event_duration*rawdata_sampling_rate-outer_trim_factor)*factor:
                print(f"____ shot {cls.shot.name} {cls.shot.direction} the time span of the file is {shot_timespan} seconds___")
                require_interpolation = True
                with open("problematic shots.txt", 'a') as file1:
                    file1.write(f"\nshot {cls.shot.name} {cls.shot.direction} the time span of the file is {shot_timespan} seconds")
                return
            
            if convert_to_phase or convert_to_strain:
                assert rawdata_to_phase_factor != None, "Can't convert the data without the rawdata_to_phase_factor. This value can be found in the H5 header info"
                cls.phase_from_raw_data(cls, rawdata_to_phase_factor=rawdata_to_phase_factor) 

            if lowpass_filter:
                cls.lowpass_filter(cls,lowpass_frequency,filter_order = lowpass_filter_order)
            
            if highpass_filter:
                cls.highpass_filter(cls,highpass_frequency,filter_order = highpass_filter_order)
                
            if downsample:
                cls.downsample(cls,new_sampling_rate)
            
            if convert_to_strain:
                cls.strain_from_raw_data(cls, laser_wavelength=laser_wavelength,fibre_refractive_index=fibre_refractive_index,
                                             gauge_length=gauge_length,rawdata_to_phase_factor=rawdata_to_phase_factor, Photoelastic_Scaling_Factor = 0.78)


                                             
                
            return cls(cls.shot,cls.event_duration,cls.no_of_channels,cls.sampling_datetime,cls.data,cls.sampling_rate,cls.resampled_from, cls.data_type,
                       cls.spatial_sampling_interval,cls.gauge_length,cls.lowpass_filtered ,cls.lowpass_frequency,cls.highpass_filtered,
                       cls.highpass_frequency, cls.downsampled, cls.datetime_description)
        
        
    def plot_waterfall(self,from_channel = 0, to_channel = None):
        """Returns a waterfall plot 

        Args:
            from_channel (int, optional): start the water fall plot from this channel. Defaults to 0.
            to_channel (_type_, optional): plot the waterfall plot up to this channel. Defaults to None.
        """
        if to_channel == None:
            print(self.no_of_channels)
            to_channel = self.no_of_channels
            
        fig, ax = plt.subplots(figsize=(10, 6),dpi=150)
        max_amp = np.max(self.data)
        
        #if an outer trim is used the number of data points is larger by 1 sample than the event duration
        time =  np.linspace(0,self.event_duration,num = int(self.data.shape[1]))
        for k in range(from_channel,to_channel):
            ax.plot((self.data[k]/max_amp)+k,time,linewidth=0.5,color='b')
            ax.set_ylim(self.event_duration,0)
            ax.yaxis.grid(linestyle=':')
            ax.set_ylabel("Time(s)")
            ax.set_xlabel("Channel")
    
    
    
    
    
    def index(self, considered_range = (), plot_range =()):
        """_finds the channel(s) with the highest positive and/or negative values and plots waterfall plot_

        Args:
            considered_range (tuple, optional): The channels considered in the indexing. Defaults to (). This means all channels will be considered.
            plot_range (tuple, optional): The range of the channels displayed in the waterfall plot. Defaults to (). Three channels before and after the channel with the largest positive or negative value.
        """

        if considered_range == ():
            from_channel = 0
            to_channel = self.no_of_channels
        else:
            assert considered_range[1] < self.no_of_channels, "the channel range can't exceed the number of the cables channels"
            from_channel = considered_range[0]
            to_channel = considered_range[1]
            
        max_value = np.amax(self.data[from_channel:to_channel,:])
        min_value = np.amin(self.data[from_channel:to_channel,:])
        
        result_max = np.where(self.data[from_channel:to_channel,:] == max_value)
        result_min = np.where(self.data[from_channel:to_channel,:] == min_value)
        if result_max[0] == result_min[0]:
            print(f"You're probably looking for channel {result_max[0]}. I would recommend that you double-check this.")
        if result_max[0] != result_min[0]:
            print(f"It's probably either channel {result_max[0]} or channel {result_min[0]}. I would recommend that you double-check this.") 
        
        if max_value > abs(min_value):
            result = int(result_max[0])
        else:
            result = int(result_min[0])
        
        fig, ax = plt.subplots(figsize=(10, 6),dpi=150)
        max_amp = np.max(self.data[from_channel:to_channel,:])
        
        time =  np.linspace(0,self.event_duration,num = int(self.data.shape[1]))
            
            
        if plot_range == ():
            for k in range(result-3,result+4):
                ax.plot((self.data[k]/max_amp)+k,time,linewidth=0.5,color='b')
                ax.set_ylim(self.event_duration,0)
                ax.yaxis.grid(linestyle=':')
                ax.set_ylabel("Time (s)")
                ax.set_xlabel("Channel")
        else:
            for k in range(plot_range[0],plot_range[1]):
                ax.plot((self.data[k]/max_amp)+k,time,linewidth=0.5,color='b')
                ax.set_ylim(self.event_duration,0)
                ax.yaxis.grid(linestyle=':')
                ax.set_ylabel("Time (s)")
                ax.set_xlabel("Channel")
    
    
    
    
    
    
    
    def plot_channel(self, channel_number):
        """Plot a specific channel waveforms

        Args:
            channel_number (int): the channel number to plot
        """
        
        time =  np.linspace(0,self.event_duration,num = int(self.data.shape[1]))
                      
        fig, ax = plt.subplots(figsize=(10, 5),dpi=150)
        ax.plot(time,self.data[channel_number])
        ax.set_ylabel(f"{self.data_type}")
        ax.set_xlabel("Time(s)")
    
    def plot_magnitude_spectrum(self, channel_number, frequency_range = (), scale = 'default'):
        """Returns the Fourier magnitude spectrum of a given channel

        Args:
            channel_number (int): the channel number to process
            frequency_range (tuple, optional): plot the Fourier magnitude spectrum between those two frequencies. Defaults to Defaults to from the 0 to Nyquist.
            scale (str, optional): can be 'dB' for decibel of default. Defaults to 'default'.
        """
        fig, ax = plt.subplots(figsize=(6, 3),dpi=150)
        ax.magnitude_spectrum(self.data[channel_number],Fs = self.sampling_rate, scale= scale)
        if frequency_range == ():
            pass
        else:
            ax.set_xlim(frequency_range[0],frequency_range[1])
            
    def plot_phase_spectrum(self, channel_number, frequency_range = (), phase_range = ()):
        """Returns the Fourier unwrapped phase spectrum of a given channel

        Args:
            channel_number (int): the channel number to process
            frequency_range (tuple, optional): plot the Fourier phase spectrum between those two frequencies. Defaults to Defaults to from the 0 to Nyquist.
            phase_range (tuple, optional): plot the Fourier phase spectrum between those two phase values. Defaults to from min to max.
        """
        fig, ax = plt.subplots(figsize=(6, 3),dpi=150)
        ax.phase_spectrum(self.data[channel_number],self.sampling_rate)
        if frequency_range == ():
            pass
        else:
            ax.set_xlim(frequency_range[0],frequency_range[1])
        
        if phase_range == ():
            pass
        else:
            ax.set_ylim(phase_range[0],phase_range[1])
            
            
    def plot_angle_spectrum(self, channel_number, frequency_range = (), phase_range = ()):
        """Returns the wrapped phase spectrum

        Args:
            channel_number (int): the channel number to process
            frequency_range (tuple, optional): plot the angle spectrum between those two frequencies.Defaults to Defaults to from the 0 to Nyquist.
            phase_range (tuple, optional): plot the angle spectrum between those two angle values. Defaults to from min to max.
        """
        fig, ax = plt.subplots(figsize=(6, 3),dpi=150)
        ax.angle_spectrum(self.data[channel_number],self.sampling_rate)
        if frequency_range == ():
            pass
        else:
            ax.set_xlim(frequency_range[0],frequency_range[1])
        
        if phase_range == ():
            pass
        else:
            ax.set_ylim(phase_range[0],phase_range[1])

        
            
    def to_csv(self, directory):
        """save event to csv file

        Args:
            directory (str): output directory
        """
        
        OUTPUT_DIR = pathlib.Path(directory)
        assert OUTPUT_DIR.exists(),"The provided directory doesn't exist"
        
        metadata = [['name',self.shot.name],['source',self.shot.source],['source_function',self.shot.config],['direction',self.shot.direction],
            ['source_Lat',self.shot.lat],['source_Lon',self.shot.lon],['start_datetime',str(self.shot.timestamp)],
            ['duration (s)',self.event_duration],['no_of_channels',self.no_of_channels],
            ['spatial_sampling_interval',self.spatial_sampling_interval],['gauge_length',self.gauge_length],
            ['sampling_rate',self.sampling_rate],['','']
           ]
        
        df = pd.DataFrame(metadata, columns=['datetime (UNIX)',0])
        
        df1 = pd.DataFrame(np.arange(0,self.no_of_channels)).T
        df1.insert(0,'datetime (UNIX)','datetime (UNIX)')
           
        df2 = pd.DataFrame(np.transpose(self.data), columns=[np.arange(0,self.no_of_channels)])
        df2 = pd.DataFrame(np.transpose(self.data), columns=None)
        df2.insert(0,'datetime (UNIX)',np.transpose(self.sampling_datetime).tolist())
        
        df3 = pd.concat([df,df1,df2])
        
        output_name = f'{directory}/{self.shot.direction}_{self.shot.name}_{self.shot.timestamp.strftime("%Y%m%d%H%M%S.%f")}.csv'
        df3.to_csv(output_name, header = False, index = False)






    def to_excel(self, directory):
        """save event to csv file

        Args:
            directory (str): output directory
        """
        
        OUTPUT_DIR = pathlib.Path(directory)
        assert OUTPUT_DIR.exists(),"The provided directory doesn't exist"
        
        source_metadata = [['name',self.shot.name],['source',self.shot.source],['source_function',self.shot.config],['direction',self.shot.direction], ['mode',self.shot.mode],
            ['lat',self.shot.lat],['lon',self.shot.lon],['local_x',self.shot.local_x],['local_y',self.shot.local_y],['start_datetime',str(self.shot.timestamp)],
            ['duration (s)',self.event_duration],['no_of_channels',self.no_of_channels],['sampling_rate',self.shot.sampling_rate],['ch1_description',self.shot.ch1_description],
            ['ch2_description',self.shot.ch2_description],['ch3_description',self.shot.ch3_description],['ch4_description',self.shot.ch4_description],['','']
           ]
        
        df = pd.DataFrame(source_metadata, columns=['ch1',0])
        
        df1 = pd.DataFrame(['ch2',"ch3","ch4"]).T
        df1.insert(0,'ch1','ch1')

           
        df2 = pd.DataFrame(data=np.hstack((self.shot.ch2[:,None],self.shot.ch3[:,None],self.shot.ch4[:,None])), columns=None) #["ch1","ch2","ch3","ch4"]
        df2.insert(0,'ch1',np.transpose(self.shot.ch1).tolist())
        
        df3 = pd.concat([df,df1,df2])
        
        
        # df3.to_excel(output_name, sheet_name = 'Source', header = False, index = False)




        DAS_metadata = [
            ['duration (s)',self.event_duration],['no_of_channels',self.no_of_channels],
            ['spatial_sampling_interval',self.spatial_sampling_interval],['gauge_length',self.gauge_length],
            ['sampling_rate',self.sampling_rate],['lowpass_filtered',self.lowpass_filtered],
            ['lowpass_frequency',self.lowpass_frequency],['highpass_filtered',self.highpass_filtered],['highpass_frequency',self.highpass_frequency],
            ['downsampled',self.downsampled],['resampled_from(Hz)',self.resampled_from],['datetime_description',self.datetime_description],['data_type',self.data_type],['','']
           ]


     
        df4 = pd.DataFrame(DAS_metadata, columns=['datetime (UNIX)',0])
        
        df5 = pd.DataFrame(np.arange(0,self.no_of_channels)).T
        df5.insert(0,'datetime (UNIX)','datetime (UNIX)')
           
        df6 = pd.DataFrame(np.transpose(self.data), columns=[np.arange(0,self.no_of_channels)])
        df6 = pd.DataFrame(np.transpose(self.data), columns=None)
        df6.insert(0,'datetime (UNIX)',np.transpose(self.sampling_datetime).tolist())
        
        df7 = pd.concat([df4,df5,df6])
        
        output_name = f'{directory}/{self.shot.direction}_{self.shot.name}_{self.shot.timestamp.strftime("%Y%m%d%H%M%S.%f")}.xlsx'
        with pd.ExcelWriter(output_name) as writer:  
            df3.to_excel(writer, sheet_name = 'Source', header = False, index = False)
            df7.to_excel(writer, sheet_name = 'DAS', header = False, index = False)

        # df7.to_excel(output_name, sheet_name = 'DAS', header = False, index = False)       
        
        
        
    def to_h5(self, directory):
        """save event to h5 file

        Args:
            directory (str): output directory 
        """
        
        OUTPUT_DIR = pathlib.Path(directory)
        assert OUTPUT_DIR.exists(),"The provided directory doesn't exist"
        output_name = f'{directory}/{self.shot.direction}_{self.shot.name}_{datetime.datetime.utcfromtimestamp(self.sampling_datetime[0]/1000000).strftime("%Y%m%d%H%M%S.%f")[:-3]}.h5'
        with h5py.File(output_name,"w") as f:
            group = f.create_group('Event')
            group.attrs['name'] = str(f"{self.shot.direction}_{self.shot.name}")
            group.attrs['duration(s)'] = str(self.event_duration)


            source_subgroup = group.create_group('Source')
            channel1 = source_subgroup.create_dataset('Ch1',data=self.shot.ch1)
            channel2 =source_subgroup.create_dataset('Ch2',data=self.shot.ch2)
            channel3 =source_subgroup.create_dataset('Ch3',data=self.shot.ch3)
            channel4 =source_subgroup.create_dataset('Ch4',data=self.shot.ch4)
            channel1.attrs['ch1_description'] = str(self.shot.ch1_description)
            channel2.attrs['ch2_description'] = str(self.shot.ch2_description)
            channel3.attrs['ch3_description'] = str(self.shot.ch3_description)
            channel4.attrs['ch4_description'] = str(self.shot.ch4_description)
            source_subgroup.attrs['name'] = str(self.shot.name)
            source_subgroup.attrs['source'] = str(self.shot.source)
            source_subgroup.attrs['start_datetime'] = str(self.shot.timestamp)           
            source_subgroup.attrs['sampling_rate(Hz)'] = str(self.shot.sampling_rate)
            source_subgroup.attrs['direction'] = str(self.shot.direction)
            source_subgroup.attrs['lat'] = str(self.shot.lat)
            source_subgroup.attrs['lon'] = str(self.shot.lon)
            source_subgroup.attrs['local_x'] = str(self.shot.local_x)
            source_subgroup.attrs['local_y'] = str(self.shot.local_y)
            source_subgroup.attrs['mode'] = str(self.shot.mode)
            source_subgroup.attrs['source_function'] = str(self.shot.config)



   
            DAS_subgroup = group.create_group('DAS')
            channels_Data = DAS_subgroup.create_dataset('Data',data=self.data)
            channels_Data.attrs['no_of_channels'] = str(self.no_of_channels)
            Absolute_time = DAS_subgroup.create_dataset('Absolute_time',data=self.sampling_datetime)
            Absolute_time.attrs['description'] = str(self.datetime_description)
            DAS_subgroup.attrs['no_of_channels'] = str(self.no_of_channels)
            DAS_subgroup.attrs['sampling_rate(Hz)'] = str(self.sampling_rate)
            DAS_subgroup.attrs['data_type'] = self.data_type
            DAS_subgroup.attrs['spatial_sampling_interval(m)'] = str(self.spatial_sampling_interval)
            DAS_subgroup.attrs['gauge_length(m)'] = str(self.gauge_length)
            DAS_subgroup.attrs['lowpass_filtered'] = str(self.lowpass_filtered)
            DAS_subgroup.attrs['lowpass_frequency'] = str(self.lowpass_frequency)
            DAS_subgroup.attrs['highpass_filtered'] = str(self.highpass_filtered)
            DAS_subgroup.attrs['highpass_frequency'] = str(self.highpass_frequency)
            DAS_subgroup.attrs['downsampled'] = str(self.downsampled)
            DAS_subgroup.attrs['resampled_from(Hz)'] = str(self.resampled_from)


    @classmethod
    def from_h5 (cls, file_path):
        """ create event object from an h5 event file saved using daspy
            Args:
            file_path (str): the path the event h5 file  
        """
        
        FILE_PATH = pathlib.Path(file_path)
        assert FILE_PATH.exists(),"The provided file doesn't exist"
        
        with h5py.File(file_path,"r") as f:
            shot_name = f['Event/Source'].attrs['name']
            event_start_time = dt.datetime.strptime(f['Event/Source'].attrs['start_datetime'],'%Y-%m-%d %H:%M:%S.%f')
            shot_mode = f['Event/Source'].attrs['mode']
            shot_direction = f['Event/Source'].attrs['direction']
            shot_lat = f['Event/Source'].attrs['lat']
            shot_lon = f['Event/Source'].attrs['lon']
            shot_source = f['Event/Source'].attrs['source']
            shot_config = f['Event/Source'].attrs['source_function']
            shot_local_x = f['Event/Source'].attrs['local_x']
            shot_local_y = f['Event/Source'].attrs['local_y']
            shot_sampling_rate = f['Event/Source'].attrs['sampling_rate(Hz)']
            ch1_description = f['Event/Source/Ch1'].attrs['ch1_description']
            ch2_description = f['Event/Source/Ch2'].attrs['ch2_description']
            ch3_description = f['Event/Source/Ch3'].attrs['ch3_description']
            ch4_description = f['Event/Source/Ch4'].attrs['ch4_description']
            ch1 = f["Event/Source/Ch1"][:]
            ch2 = f["Event/Source/Ch2"][:]
            ch3 = f["Event/Source/Ch3"][:]
            ch4 = f["Event/Source/Ch4"][:]


            cls.event_duration = float(f['Event'].attrs['duration(s)']) 
            cls.no_of_channels = int(f['Event/DAS'].attrs['no_of_channels']) 
            cls.sampling_rate = int(f['Event/DAS'].attrs['sampling_rate(Hz)']) 
            cls.data_type = f['Event/DAS'].attrs['data_type'] 
            cls.spatial_sampling_interval = float(f['Event/DAS'].attrs['spatial_sampling_interval(m)']) 
            cls.gauge_length = float(f['Event/DAS'].attrs['gauge_length(m)']) 
            cls.lowpass_filtered = bool(f['Event/DAS'].attrs['lowpass_filtered']) 
            cls.lowpass_frequency = float(f['Event/DAS'].attrs['lowpass_frequency']) 
            cls.highpass_filtered = bool(f['Event/DAS'].attrs['highpass_filtered']) 
            cls.highpass_frequency = float(f['Event/DAS'].attrs['highpass_frequency']) 
            cls.downsampled = bool(f['Event/DAS'].attrs['downsampled'])
            cls.resampled_from = int(f['Event/DAS'].attrs['resampled_from(Hz)'])
            cls.resampled_from = int(f['Event/DAS'].attrs['resampled_from(Hz)'])
            cls.datetime_description = str(f["Event/DAS/Absolute_time"].attrs['description'])
            cls.sampling_datetime = f["Event/DAS/Absolute_time"][:]
            cls.data= f["Event/DAS/Data"][:,:]

        
        cls.shot = Shot(name = shot_name, timestamp=event_start_time, mode = shot_mode, direction = shot_direction, lat = shot_lat, lon= shot_lon, source=shot_source, config = shot_config,
        local_x=shot_local_x, local_y=shot_local_y, sampling_rate=shot_sampling_rate, ch1_description=ch1_description, ch1=ch1, ch2_description=ch2_description, ch2=ch2, ch3_description=ch3_description,
        ch3=ch3, ch4=ch4, ch4_description=ch4_description)

        return cls(cls.shot,cls.event_duration,cls.no_of_channels,cls.sampling_datetime,cls.data,cls.sampling_rate, cls.data_type,
                       cls.spatial_sampling_interval,cls.gauge_length,cls.lowpass_filtered ,cls.lowpass_frequency,cls.highpass_filtered,
                       cls.highpass_frequency, cls.downsampled)



        
    def temporal_upsample (self, new_sampling_rate: int, kind = 'linear'): #,start = 10, finish = 20,channel_no =0):
        """Increases the time domain sampling frequency using the provided interpolation function 

        Args:
            new_sampling_rate (int): _The new sampling frequency in Hz. It has to be larger than the existing sampling rate_.
            kind (str, optional): _The interpolation function used to upsample the time series_. available options are ('linear', 'nearest', 'nearest-up', 'zero', 'slinear', 'quadratic', 'cubic', 'previous', or 'next'. 'zero', 'slinear', 'quadratic' and 'cubic' ). Defaults to 'linear'.
        Returns:
            _Event object_: _The object is returned with updated data, sampling_datetime, and sampling_rate_
        """

        assert new_sampling_rate > self.sampling_rate, "The provided sampling rate is lesser than or equal to the existing sampling rate. You may try the downsample function"
        


        if self.sampling_datetime.shape[0] > self.sampling_rate*self.event_duration:
            sampling_datetime =  np.linspace(self.sampling_datetime[0],self.sampling_datetime[self.sampling_datetime.shape[0]-1],num = int(new_sampling_rate*self.event_duration+(new_sampling_rate/self.sampling_rate)))
        else:
            sampling_datetime =  np.linspace(self.sampling_datetime[0],self.sampling_datetime[self.sampling_datetime.shape[0]-1],num = int(new_sampling_rate*self.event_duration)) 

        
        data_list = []
        for trace_no in range(self.data.shape[0]):
             f = interpolate.interp1d(self.sampling_datetime, self.data[trace_no], kind= kind, fill_value= "nan")
             data = f(sampling_datetime)
             data_list.append(data)


        # factor = int(new_sampling_rate/self.sampling_rate)
        # plt.figure(figsize=(10, 5))
        # plt.plot(self.sampling_datetime[start:finish], self.data[channel_no][start:finish], 'o', sampling_datetime[start*factor:finish*factor], data_list[channel_no][start*factor:finish*factor], '-')

        if self.sampling_datetime.shape[0] > self.sampling_rate*self.event_duration:
            self.sampling_datetime = np.empty((int(self.event_duration*new_sampling_rate+(new_sampling_rate/self.sampling_rate))), dtype = np.int64)
            self.data = np.empty((self.no_of_channels, int(self.event_duration*new_sampling_rate+(new_sampling_rate/self.sampling_rate))), dtype = np.int32)
        else:
            self.sampling_datetime = np.empty((int(self.event_duration*new_sampling_rate+(new_sampling_rate/self.sampling_rate))), dtype = np.int64)
            self.data = np.empty((self.no_of_channels, int(self.event_duration*new_sampling_rate+(new_sampling_rate/self.sampling_rate))), dtype = np.int32)
        
        self.resampled_from = self.sampling_rate
        self.sampling_rate = new_sampling_rate
        self.data = np.asarray(data_list)
        self.sampling_datetime = sampling_datetime

        if(np.isnan(self.data).any()):
            print("The wavefield contain nan values")
        
        return self


      
    
    def spatial_resample (self, number_of_channels: int, kind = 'linear'):
        """_Changes the spatial sampling interval (i.e., number of channels) using the provided interpolation function_

        Args:
            number_of_channels (_int_): _The number of channels that satisfies the required sampling intervals_
            kind (str, optional): The interpolation function used for spatial upsampling_. available options are ('linear', 'cubic', and 'quintic'). Defaults to 'linear'.

        Returns:
            _Event object_: _The object is returned with updated data, no_of_channels, and spatial_sampling_interval
        """
          

        distance = self.no_of_channels * self.spatial_sampling_interval
        existing_sampling_interval =  np.linspace(0,distance,num = self.no_of_channels)
        new_sampling_interval =  np.linspace(0,distance,num = number_of_channels)
        
        f = interpolate.interp2d(self.sampling_datetime,existing_sampling_interval,self.data, kind= kind, fill_value= "nan")
        data = f(self.sampling_datetime,new_sampling_interval)
        # plt.figure(figsize=(10, 5))
        # plt.plot(existing_sampling_interval[start:finish], self.data[start:finish,time_step], 'o', new_sampling_interval[start:finish+15], data[start:finish+15,time_step], '*')
        # plt.plot(new_sampling_interval[start:finish+15], data[start:finish+15,time_step], '-')
        
        self.data = np.empty((number_of_channels, self.data.shape[1]), dtype = np.int32)
        self.no_of_channels = number_of_channels
        self.spatial_sampling_interval = distance/number_of_channels
        self.data = data


        if(np.isnan(self.data).any()):
            print("The wavefield contain nan values")
        
        return self
    
    
    
    
        # def temporal_upsample (self, new_sampling_rate, kind = 'linear'): #,start = 10, finish = 20,channel_no =0):
    #     """Increases the time domain sampling frequency using the provided interpolation function 

    #     Args:
    #         new_sampling_rate (int): _The new sampling frequency in Hz. It has to be larger than the existing sampling rate_.
    #         kind (str, optional): _The interpolation function used to upsample the time series_. available options are ('linear', 'cubic', and 'quintic'). Defaults to 'linear'.
    #     Returns:
    #         _Event object_: _The object is returned with updated data, sampling_datetime, and sampling_rate_
    #     """

    #     assert new_sampling_rate > self.sampling_rate, "The provided sampling rate is lesser than or equal to the existing sampling rate. You may try the downsample function"
        

    #     channels =  np.arange(0,self.no_of_channels,1)
        
    #     if self.sampling_datetime.shape[0] > self.sampling_rate*self.event_duration:
    #         sampling_datetime =  np.linspace(self.sampling_datetime[0],self.sampling_datetime[self.sampling_datetime.shape[0]-1],num = int(new_sampling_rate*self.event_duration+(new_sampling_rate/self.sampling_rate)))
    #     else:
    #         sampling_datetime =  np.linspace(self.sampling_datetime[0],self.sampling_datetime[self.sampling_datetime.shape[0]-1],num = int(new_sampling_rate*self.event_duration)) 

    #     f = interpolate.interp2d(self.sampling_datetime,channels,self.data, kind= kind, fill_value= "nan")
    #     data = f(sampling_datetime,channels)
        
    #     # factor = int(new_sampling_rate/self.sampling_rate)
    #     # plt.figure(figsize=(10, 5))
    #     # plt.plot(self.sampling_datetime[start:finish], self.data[channel_no][start:finish], 'o', sampling_datetime[start*factor:finish*factor], data[channel_no][start*factor:finish*factor], '-')

    #     if self.sampling_datetime.shape[0] > self.sampling_rate*self.event_duration:
    #         self.sampling_datetime = np.empty((int(self.event_duration*new_sampling_rate+(new_sampling_rate/self.sampling_rate))), dtype = np.int64)
    #         self.data = np.empty((self.no_of_channels, int(self.event_duration*new_sampling_rate+(new_sampling_rate/self.sampling_rate))), dtype = np.int32)
    #     else:
    #         self.sampling_datetime = np.empty((int(self.event_duration*new_sampling_rate+(new_sampling_rate/self.sampling_rate))), dtype = np.int64)
    #         self.data = np.empty((self.no_of_channels, int(self.event_duration*new_sampling_rate+(new_sampling_rate/self.sampling_rate))), dtype = np.int32)

    #     self.sampling_rate = new_sampling_rate
    #     self.data = data
    #     self.sampling_datetime = sampling_datetime

    #     if(np.isnan(self.data).any()):
    #         print("The wavefield contain nan values")
        
    #     return self