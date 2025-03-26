"""
This module provides functionalities for interacting with audio files and
storing data efficiently. It includes utilities for extracting metadata from
audio files, such as duration and timestamps, and for saving pandas DataFrames
to disk in specified formats.

The module uses external libraries like `audioread` for audio file processing
and `pandas` for data manipulation, ensuring wide compatibility and ease of
integration into data processing workflows.

Functions:
- get_audio_info(audio_path, format_name, date_time_func=None,
  format_file_path=None, store_duration=0, perc_sample=1): Extracts information
  from audio files, returning a DataFrame with details like timestamps and
  duration.
- store_df(df, file_type, base_dir, file_name): Saves a DataFrame to disk in a
  specified format (CSV or Pickle), facilitating data persistence and sharing.

Examples and detailed parameter information are provided within each function's
docstring, guiding usage and application in various scenarios.

Note:
- This module is part of the `maui` package, focusing on audio file analysis and
  data management.

Dependencies:
- pandas: For DataFrame operations.
- audioread: For accessing audio file information.
- glob, os, datetime, random: For file and directory operations, and handling
  dates and randomness.
"""

import random
import os
import glob

import audioread

import pandas as pd

# maui imports
import maui.files_metadata


# ------------------------------------------------


def get_audio_info(
    audio_path,
    format_name,
    date_time_func=None,
    format_file_path=None,
    store_duration=False,
    perc_sample=1,
):
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

        >>> audio_dir = "/path/to/audio/directory"
        >>> df = io.get_audio_info(audio_dir, "LEEC_FILE_FORMAT",
                                   store_duration=True, perc_sample=1)
        >>> df["dt"] = pd.to_datetime(df["timestamp_init"]).dt.date
    """

    file_dict = None

    if os.path.isfile(audio_path):
        basename = os.path.basename(audio_path)
        filename, _ = os.path.splitext(basename)

        file_dict = maui.files_metadata.extract_metadata(
            filename, format_name, date_time_func, format_file_path
        )

        if store_duration:
            # Usa o with para garantir o fechamento do arquivo após acessar a duração
            with audioread.audio_open(audio_path) as x:
                duration = x.duration
            file_dict["duration"] = duration

        file_dict["file_path"] = audio_path
        df = pd.DataFrame(file_dict, index=[0])

    elif os.path.isdir(audio_path):
        file_dict = []

        for file_path in glob.glob(os.path.join(audio_path, "*.wav")):
            if random.uniform(0, 1) < perc_sample:
                basename = os.path.basename(file_path)
                filename, _ = os.path.splitext(basename)

                file_dict_temp = maui.files_metadata.extract_metadata(
                    filename, format_name, date_time_func, format_file_path
                )

                if store_duration:
                    # Usa o with para garantir o fechamento do arquivo
                    with audioread.audio_open(file_path) as x:
                        duration = x.duration
                    file_dict_temp["duration"] = duration

                file_dict_temp["file_path"] = file_path
                file_dict.append(file_dict_temp)

        df = pd.DataFrame(file_dict)

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

    available_file_types = ["csv", "pickle"]

    if file_type not in available_file_types:
        raise ValueError("File type not available")

    if file_type == "csv":
        full_path = os.path.join(base_dir, file_name + ".csv")
        df.to_csv(full_path)

        return

    if file_type == "pickle":
        full_path = os.path.join(base_dir, file_name + ".pkl")
        df.to_pickle(full_path)

        return
