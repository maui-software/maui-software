import librosa
import librosa.display
import audioread

import numpy as np
import pandas as pd
import random


import os
import glob

import time
import datetime

def get_file_structure_leec(filename):

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
    
    if (file_type == 'csv'):
        full_path = os.path.join(base_dir, file_name+'.csv')
        df.to_csv(full_path)
        
        return
    
    elif (file_type == 'pickle'):
        full_path = os.path.join(base_dir, file_name+'.pkl')
        df.to_pickle(full_path)
        
        return
    