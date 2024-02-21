import audioread

import pandas as pd
import random


import os
import glob

import time
import datetime

# maui imports
import maui.files_metadata


# ------------------------------------------------

def get_audio_info(audio_path, format_name, date_time_func=None, format_file_path=None, store_duration=0, perc_sample=1):
    """
    Extract audio file information from a file or directory.

    This function processes audio files specified by the 'audio_path' argument,
    extracting information such as filename structure, timestamps, and duration.
    It can handle both single audio files and entire directories of audio files.

    Parameters
    ----------
        audio_path: str
            The path to an audio file or directory containing audio files.
        store_duration: int, optional
            Whether to calculate and store audio duration (default is 0).
        perc_sample float, optional
            Percentage of audio files to include when processing a directory (default is 1).

    Returns
    -------
        df: pandas.DataFrame
            A DataFrame containing information about the audio files.

    Raises
    ------
        Exception: 
            If the input is neither a file nor a directory.

    Examples
    --------
        >>> from maui import io
        >>> audio_file = "forest_channelA_20210911_153000_jungle.wav"
        >>> io.get_audio_info(audio_file, store_duration=1, perc_sample=0.8)
           landscape   channel      date    time environment      timestamp_init timestamp_end  duration                                   file_path
        0     forest  channelA  20210911  153000      jungle 2021-09-11 15:30:00          None       NaN  forest_channelA_20210911_153000_jungle.wav

        >>> audio_dir = "/path/to/audio/directory"
        >>> io.get_audio_info(audio_dir, store_duration=0, perc_sample=0.5)
           landscape   channel      date    time environment      timestamp_init          file_path
        0     forest  channelA  20210911  153000      jungle 2021-09-11 15:30:00  /path/to/audio/directory/forest_channelA_20210911_153000_jungle.wav
        1   mountains  channelB  20210911  160000      forest 2021-09-11 16:00:00  /path/to/audio/directory/mountains_channelB_20210911_160000_forest.wav
    
    """


    file_dict = None

    if os.path.isfile(audio_path):
        basename = os.path.basename(audio_path)
        filename, file_extension = os.path.splitext(basename)

        file_dict = maui.files_metadata.extract_metadata(filename, format_name, date_time_func, format_file_path)
        
        file_dict['timestamp_end'] =  None
        file_dict['duration'] =  None
        
        if (store_duration):
            x = audioread.audio_open(audio_file)
            duration = x.duration
            
            file_dict['timestamp_end'] =  file_dict['timestamp_init'] + datetime.timedelta(seconds=duration)
            file_dict['duration'] =  duration
        
        file_dict['file_path'] = audio_file
        
        df = pd.DataFrame(file_dict, index=[0])

    elif os.path.isdir(audio_path):
        file_dict = []
        for file_path in glob.glob(audio_path + '/*.wav'):

            if random.uniform(0, 1) < perc_sample:
                basename = os.path.basename(file_path)
                filename, file_extension = os.path.splitext(basename)

                file_dict_temp = maui.files_metadata.extract_metadata(filename, format_name, date_time_func, format_file_path)

                file_dict_temp['timestamp_end'] =  None
                file_dict_temp['duration'] =  None

                if (store_duration):
                    x = audioread.audio_open(file_path)
                    duration = x.duration

                    file_dict_temp['timestamp_end'] =  file_dict_temp['timestamp_init'] + datetime.timedelta(seconds=duration)
                    file_dict_temp['duration'] =  duration
                
                file_dict_temp['file_path'] = file_path

                file_dict.append(file_dict_temp)
        
        df = pd.DataFrame(file_dict)

        df['hour'] = pd.to_datetime(df['timestamp_init']).dt.hour
        df['time'] = pd.to_datetime(df['timestamp_init']).dt.time
        df['dt'] = pd.to_datetime(df['timestamp_init']).dt.date
        
    else:
        raise Exception("The input must be a file or a directory")

    return df

# ------------------------------------------------

def store_df(df, file_type, base_dir, file_name):
    """
    Store a DataFrame to a file in a specified format.

    This function takes a DataFrame 'df' and saves it to a file in the specified
    'file_type' and location, combining 'base_dir' and 'file_name'.

    Parameters
    ----------
        df: pandas.DataFrame
            The DataFrame to be saved.
        file_type: str
            The file format to use for storing the DataFrame ('csv' or 'pickle').
        base_dir: str
            The base directory where the file will be saved.
        file_name: str
            The name of the file (without file extension).

    Returns
    -------
        None

    Examples
    --------
        >>> from maui import io
        >>> data = {'A': [1, 2, 3], 'B': ['a', 'b', 'c']}
        >>> df = pd.DataFrame(data)
        >>> io.store_df(df, 'csv', '/path/to/directory', 'my_dataframe')
        # Saves the DataFrame as '/path/to/directory/my_dataframe.csv'

        >>> io.store_df(df, 'pickle', '/path/to/directory', 'my_dataframe')
        # Saves the DataFrame as '/path/to/directory/my_dataframe.pkl'

    """
    
    if (file_type == 'csv'):
        full_path = os.path.join(base_dir, file_name+'.csv')
        df.to_csv(full_path)
        
        return
    
    elif (file_type == 'pickle'):
        full_path = os.path.join(base_dir, file_name+'.pkl')
        df.to_pickle(full_path)
        
        return
    