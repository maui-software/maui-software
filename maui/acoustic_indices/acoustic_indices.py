"""
Module for Calculating Acoustic Indices from Audio Files

This module provides functions to calculate acoustic indices from audio files stored 
in a pandas DataFrame. It supports both parallel and sequential processing, 
dividing the DataFrame into smaller chunks to optimize performance.

Functions
---------
- calculate_acoustic_indices(df_init, file_path_col, acoustic_indices_methods, 
  pre_calculation_method, parallel, chunk_size=None, temp_dir='./tmp_maui_ac_files/'): 
  Calculates acoustic indices for audio files in a DataFrame, with support for 
  parallel processing.

Dependencies
------------
- numpy
- pandas
- maad
- tqdm
- tempfile
- os
- gc
- functools.partial
- multiprocessing as mp

"""

import gc
import os
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd

from maad import sound

def _convert_if_list_string(value):
    """
    Helper function to convert a space-separated or comma-separated string of floats
    or integers into a list of floats or integers.

    Parameters:
    value (any): The value to check and possibly convert.

    Returns:
    any: The original value or a converted list of floats or integers if the string
    represents a list of numbers.
    """
    if isinstance(value, str):
        value = value.strip("[]")  # Remove the brackets
        value = value.replace("\n", " ")  # Replace newlines with spaces
        value = value.replace(",", " ")  # Replace commas with spaces
        try:
            # Convert the cleaned string to a list of floats or integers
            return [float(x) if "." in x else int(x) for x in value.split()]
        except ValueError:
            # If conversion fails, return the original string
            return value
    return value


def _convert_string_to_list(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert string representations of lists in a DataFrame to actual lists.

    Parameters:
    df (pd.DataFrame): The DataFrame to process.

    Returns:
    pd.DataFrame: The DataFrame with strings converted to lists where applicable.
    """
    for col in df.columns:
        if df[col].dtype == "object":  # Only process columns with object dtype
            df[col] = df[col].apply(_convert_if_list_string)
    return df


def _extract_indices_worker(
    df_chunk,
    file_path_col: str,
    acoustic_indices_methods,
    pre_calculation_method,
    temp_dir,
    **kwargs,
) -> str:
    """
    Helper function to extract acoustic indices from audio files and save them
    temporarily in a CSV file.

    This function processes a chunk of a DataFrame to calculate acoustic indices for
    audio files. It loads each audio file, applies a pre-calculation method, and then
    computes multiple acoustic indices using the provided methods. The results are stored
    in a temporary CSV file and the original data types of the DataFrame columns are retained.

    Parameters
    ----------
    df_chunk : tuple of (pd.DataFrame, int)
        A tuple containing a chunk of the DataFrame and its corresponding index.
        The DataFrame must have at least the column specified by `file_path_col`.

    file_path_col : str
        The name of the column in the DataFrame that contains the file paths to the audio files.

    acoustic_indices_methods : list of callables
        A list of methods, where each method computes a specific acoustic index.
        Each method should accept the output of `pre_calculation_method` and return a dictionary
        of index names and values.

    pre_calculation_method : callable
        A method that performs any necessary pre-calculations on the audio data.
        It should accept the loaded audio data and sampling rate and return an object
        that will be passed to each of the `acoustic_indices_methods`.

    temp_dir : str
        The directory path where the temporary CSV file will be saved.

    **kwargs : dict, optional
        Additional keyword arguments:
        - parallel (bool): If True, the function is running in parallel mode.
        - fid (str): A file identifier used when not running in parallel mode.

    Returns
    -------
    temp_file_path : str
        The file path to the temporary CSV file where the calculated indices are saved.

    original_dtypes : dict
        A dictionary mapping column names to their original data types in the DataFrame.

    Notes
    -----
    - This function assumes that the DataFrame `df_chunk` contains a column with file paths
      to the audio files. It processes each file, calculating the required indices and storing
      them in a temporary file.
    - If the audio file cannot be loaded, the function will print a message and skip the
      calculation for that file.
    - The `gc.collect()` calls are used to free memory after processing each row and after
      creating the temporary file.
    """

    indices_temp = {}

    df, fidx = df_chunk
    df = df.copy()

    for chunk_index, (_, row) in enumerate(df.iterrows()):
        s, fs = sound.load(row[file_path_col])

        if len(s) == 0:
            print(
                f"Sound loading failed or the file {row['file_path']} "\
                "is corrupted. Acoustic indices not calculated."
            )
        else:
            pre_calc_vars = pre_calculation_method(s, fs)
            for method in acoustic_indices_methods:
                indices_dict = method(pre_calc_vars)
                for key, value in indices_dict.items():
                    if key not in indices_temp:
                        indices_temp[key] = np.empty(len(df), dtype=type(value))
                        indices_temp[key][:] = np.nan  # Initialize with NaNs
                    indices_temp[key][
                        chunk_index
                    ] = value  # Use chunk_index instead of index
            del indices_dict
            gc.collect()

    if not kwargs["parallel"]:
        file_id = kwargs["fid"]
    else:
        file_id = f"{mp.current_process().pid}_{fidx}"

    temp_file_path = os.path.join(temp_dir, f"temp_{file_id}.csv")

    for key, value in indices_temp.items():
        df[key] = value
    df.to_csv(temp_file_path, index=False)
    original_dtypes = df.dtypes.to_dict()

    del indices_temp
    gc.collect()

    return temp_file_path, original_dtypes


def calculate_acoustic_indices(
    df_init: pd.DataFrame,
    file_path_col: str,
    acoustic_indices_methods: list,
    pre_calculation_method,
    parallel: bool,
    chunk_size: int = None,
    temp_dir: str = "./tmp_maui_ac_files/",
) -> pd.DataFrame:
    """
    Calculate acoustic indices for audio files in a DataFrame.

    This method processes a DataFrame containing file paths to audio files, calculates
    acoustic indices using the specified methods, and returns a DataFrame with the results.
    The calculations can be performed in parallel or sequentially, depending on the `parallel` flag.

    Parameters
    ----------
    df_init : pd.DataFrame
        The initial DataFrame containing the file paths to audio files and any other
        necessary metadata.

    file_path_col : str
        The name of the column in `df_init` that contains the file paths to the audio files.

    acoustic_indices_methods : list of callables
        A list of methods, where each method computes a specific acoustic index.
        Each method should accept the output of `pre_calculation_method` and return a dictionary
        of index names and values.

    pre_calculation_method : callable
        A method that performs any necessary pre-calculations on the audio data.
        It should accept the loaded audio data and sampling rate, returning an object
        that will be passed to each of the `acoustic_indices_methods`.

    parallel : bool
        If True, the calculations will be performed in parallel using multiple processes.
        If False, the calculations will be performed sequentially.

    chunk_size : int, optional
        The number of rows to process in each chunk. If not provided, a default value is calculated
        based on the number of CPU cores available.

    temp_dir : str, optional
        The directory path where the temporary CSV files will be saved.
        The default is './tmp_maui_ac_files/'.

    Returns
    -------
    pd.DataFrame
        A DataFrame containing the original data along with the calculated acoustic indices.

    Notes
    -----
    - The method first divides the DataFrame into smaller chunks, each of which is
      processed separately to calculate the acoustic indices. The results are saved
      as temporary CSV files.
    - If `parallel` is True, multiple processes are used to calculate the indices concurrently.
      Otherwise, the calculation is done sequentially.
    - The method combines the results from all chunks into a single DataFrame, restores the original
      data types, and removes the temporary files.
    - The `_convert_string_to_list` function is applied to the final DataFrame to ensure that the
      data types are correctly interpreted.

    Example
    -------
    >>> from maui import samples, utils, acoustic_indices
    >>> df = samples.get_audio_sample(dataset="leec")
    >>> df["dt"] = pd.to_datetime(df["timestamp_init"]).dt.date
    >>> def pre_calculation_method(s, fs):   
    >>>     Sxx_power, tn, fn, ext = maad.sound.spectrogram (s, fs) 
    >>>     Sxx_noNoise= maad.sound.median_equalizer(Sxx_power, display=False, extent=ext) 
    >>>     Sxx_dB_noNoise = maad.util.power2dB(Sxx_noNoise)
    >>> 
    >>>     Sxx, tn, fn, ext = maad.sound.spectrogram(s, fs, mode='amplitude')
    >>>     
    >>>     pre_calc_vars = {'Sxx': Sxx, 'tn':tn , 'fn':fn , 'ext':ext, 'Sxx_dB_noNoise':Sxx_dB_noNoise }
    >>>     return pre_calc_vars
    >>>         
    >>> def get_aci(pre_calc_vars):
    >>>     aci_xx, aci_per_bin, aci_sum  = maad.features.acoustic_complexity_index(pre_calc_vars['Sxx'])
    >>>     indices = {'aci_xx': aci_xx, 'aci_per_bin':aci_per_bin , 'aci_sum':aci_sum}
    >>>     return indices
    >>> 
    >>> def get_spectral_events(pre_calc_vars):
    >>>     EVNspFract_per_bin, EVNspMean_per_bin, EVNspCount_per_bin, EVNsp = maad.features.spectral_events(
    >>>                 pre_calc_vars['Sxx_dB_noNoise'],
    >>>                 dt=pre_calc_vars['tn'][1] - pre_calc_vars['tn'][0],
    >>>                 dB_threshold=6,
    >>>                 rejectDuration=0.1,
    >>>                 display=False,
    >>>                 extent=pre_calc_vars['ext'])  
    >>>     
    >>>     indices = {'EVNspFract_per_bin': EVNspFract_per_bin, 'EVNspMean_per_bin':EVNspMean_per_bin , 'EVNspCount_per_bin':EVNspCount_per_bin, 'EVNsp':EVNsp}
    >>>     return indices
    >>> def get_spectral_activity(pre_calc_vars):
    >>>     ACTspfract_per_bin, ACTspcount_per_bin, ACTspmean_per_bin = maad.features.spectral_activity(pre_calc_vars['Sxx_dB_noNoise'])
    >>>     indices = {'ACTspfract_per_bin': ACTspfract_per_bin, 'ACTspcount_per_bin':ACTspcount_per_bin , 'ACTspmean_per_bin':ACTspmean_per_bin}
    >>>     return indices
    >>> acoustic_indices_methods = [get_aci, get_spectral_activity, get_spectral_events]
    >>> result_df = acoustic_indices.calculate_acoustic_indices(
        df, 'file_path', indices_methods, pre_calc, parallel=True)
    """

    os.makedirs(temp_dir, exist_ok=True)

    num_processes = mp.cpu_count()

    if chunk_size is None:
        chunk_size = min(len(df_init) // num_processes + 1, 20)

    df_chunks = [
        df_init.iloc[i : i + chunk_size] for i in range(0, len(df_init), chunk_size)
    ]
    df_chunks = [(df, idx) for idx, df in enumerate(df_chunks)]

    print("Calculating acoustic indices...")

    if parallel:

        pool = mp.Pool(processes=num_processes)
        worker = partial(
            _extract_indices_worker,
            file_path_col=file_path_col,
            acoustic_indices_methods=acoustic_indices_methods,
            pre_calculation_method=pre_calculation_method,
            temp_dir=temp_dir,
            parallel=parallel,
        )

        temp_files = pool.map(worker, df_chunks)

        print("Joinning threads...")

        pool.close()
        pool.join()
    else:
        temp_files = []
        for it, df_it in enumerate(df_chunks):
            result = _extract_indices_worker(
                df_it,
                file_path_col,
                acoustic_indices_methods,
                pre_calculation_method,
                temp_dir,
                parallel=parallel,
                fid=it,
            )

            temp_files.append(result)

    print("Preparing final dataframe and removing temporary files...")

    # Combine results from temp files
    combined_df = []
    for file in temp_files:
        file, dtypes = file
        df_temp = pd.read_csv(file)
        df_temp = df_temp.astype(dtypes)
        combined_df.append(df_temp)
        os.remove(file)

    combined_df = pd.concat(combined_df, ignore_index=True)

    print("Fixing data types...")
    combined_df = _convert_string_to_list(combined_df)

    return combined_df
