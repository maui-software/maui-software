import audioread

import pandas as pd
import random


import os
import glob

import time
import datetime

def get_file_structure_leec(filename):
    """
    Parse a filename following a specific naming convention to extract relevant information.

    The function assumes that the filename follows the convention:
    'landscape_channel_date_time_environment.extension'

    Parameters:
    - filename (str): The filename to be parsed.

    Returns:
    - audio_dict (dict): A dictionary containing extracted information, including:
        - 'landscape' (str): The landscape from the filename.
        - 'channel' (str): The channel from the filename.
        - 'timestamp_init' (datetime.datetime): The date and time as a datetime object.
        - 'environment' (str): The environment from the filename.

    Example usage:
    ```
    filename = 'urban_channel01_20230910_124500_forest.wav'
    audio_info = get_file_structure_leec(filename)
    print(audio_info)
    # Output: {'landscape': 'urban', 'channel': 'channel01', 'timestamp_init': datetime.datetime(2023, 9, 10, 12, 45), 'environment': 'forest'}
    ```
    """

    file_name_segments = filename.split('_')

    try:
        while True:
            file_name_segments.remove('')
    except ValueError:
        pass

    dict_keys = ['landscape', 'channel', 'date', 'time', 'environment']
    dt_timestamp = datetime.datetime.strptime(file_name_segments[2] + ' ' + file_name_segments[3],"%Y%m%d %H%M%S")
    
    audio_dict = {
        'landscape': file_name_segments[0],
        'environment': file_name_segments[4],
        'channel': file_name_segments[1],
        'timestamp_init': dt_timestamp
    }

    return audio_dict

# ------------------------------------------------

def get_audio_info(audio_path, store_duration=0, perc_sample=1):
    """
    Retrieve audio file information from a file or a directory of audio files.

    Parameters:
    - audio_path (str): The path to an audio file or a directory containing audio files.
    - store_duration (int, optional): Whether to store audio duration information (1 for yes, 0 for no). Default is 0.
    - perc_sample (float, optional): The percentage of audio samples to process when audio_path is a directory. Default is 1.0 (100%).

    Returns:
    - df (pd.DataFrame): A DataFrame containing audio file information, including columns such as:
        - 'landscape' (str): The landscape from the filename.
        - 'channel' (str): The channel from the filename.
        - 'timestamp_init' (datetime.datetime): The date and time as a datetime object.
        - 'timestamp_end' (datetime.datetime): The end date and time as a datetime object (if store_duration is enabled).
        - 'duration' (float): The duration of the audio in seconds (if store_duration is enabled).
        - 'environment' (str): The environment from the filename.
        - 'file_path' (str): The path to the audio file.

    Example usage:
    ```
    audio_file = 'urban_channel01_20230910_124500_forest.wav'
    audio_info_df = get_audio_info(audio_file, store_duration=1)
    print(audio_info_df)
    # Output: A DataFrame with audio file information.
    ```
    """

    file_dict = None

    if os.path.isfile(audio_path):
        basename = os.path.basename(audio_path)
        filename, file_extension = os.path.splitext(basename)

        file_dict = get_file_structure_leec(filename)
        
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

                file_dict_temp = get_file_structure_leec(filename)
                
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
        
    else:
        raise Exception("The input must be a file or a directory")

    return df

# ------------------------------------------------

def store_df(df, file_type, base_dir, file_name):
    """
    Store a DataFrame to a specified file format in a given directory.

    Parameters:
    - df (pd.DataFrame): The DataFrame to be stored.
    - file_type (str): The desired file format ('csv' or 'pickle') for saving the DataFrame.
    - base_dir (str): The base directory where the file will be saved.
    - file_name (str): The desired name of the output file (without extension).

    Returns:
    - None

    Example usage:
    ```
    import pandas as pd

    # Create a sample DataFrame
    df = pd.DataFrame({'A': [1, 2, 3], 'B': [4, 5, 6]})

    # Store the DataFrame as a CSV file
    store_df(df, file_type='csv', base_dir='output_directory', file_name='my_data')

    # Store the DataFrame as a pickle file
    store_df(df, file_type='pickle', base_dir='output_directory', file_name='my_data')
    ```
    """
    
    if (file_type == 'csv'):
        full_path = os.path.join(base_dir, file_name+'.csv')
        df.to_csv(full_path)
        
        return
    
    elif (file_type == 'pickle'):
        full_path = os.path.join(base_dir, file_name+'.pkl')
        df.to_pickle(full_path)
        
        return
    