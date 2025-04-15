"""
This module provides utilities for processing and segmenting audio data, as well
as helper functions for working with other modules in the package. It includes
methods for checking time overlaps and gaps, segmenting audio files based on
duration, and preparing datasets for further analysis, such as calculating
acoustic indices.

The module handles various tasks related to audio data, including retrieving file
durations, checking for temporal inconsistencies, and managing audio segmentation.

Functions:

    segment_audio_files: Segments audio files and creates new DataFrame entries 
    for each segment.
    
    false_color_spectrogram_prepare_dataset: Prepares a dataset for false-color 
    spectrograms, checking audio duration, segmenting files, and optionally 
    calculating acoustic indices.

Dependencies
------------
- audioread: To retrieve audio file duration.
- pandas: For handling tabular data and managing timestamps.
- wave: To read and manipulate WAV audio files.
- os: To manage filesystem operations.
- datetime: To handle time-related calculations.
- maui.acoustic_indices: For calculating acoustic indices after segmentation.
"""

import os
from datetime import timedelta
import wave

import audioread
import pandas as pd
import matplotlib as mpl

from maui import acoustic_indices


def _get_audio_duration(file_path):
    """
    Retrieve the duration of an audio file in seconds.

    Parameters
    ----------
    file_path : str
        The path to the audio file.

    Returns
    -------
    float
        Duration of the audio file in seconds, or None if there is an error.
    """
    try:
        with audioread.audio_open(file_path) as f:
            return f.duration
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None


def _check_overlaps(df, datetime_col):
    """
    Check if there are overlaps between audio segments based on start and end times.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the audio segments data.
    datetime_col : str
        Column name for the start time of the segments.

    Returns
    -------
    bool
        True if overlaps are found, False otherwise.
    """
    for i in range(len(df) - 1):
        if df.loc[i, "end_time"] > df.loc[i + 1, datetime_col]:
            return True
    return False


def _check_time_gaps(df, datetime_col):
    """
    Identify time gaps between consecutive audio segments.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the audio segments data.
    datetime_col : str
        Column name for the start time of the segments.

    Returns
    -------
    list of tuples
        A list of tuples representing the time gaps between segments.
        Each tuple contains the end time of one segment and the start time of the next segment.
    """
    gaps = []
    for i in range(len(df) - 1):
        if df.loc[i, "end_time"] != df.loc[i + 1, datetime_col]:
            gaps.append((df.loc[i, "end_time"], df.loc[i + 1, datetime_col]))
    return gaps


def _unit_conversion(unit):
    """
    Convert unit identifier to its corresponding duration in seconds.

    Parameters
    ----------
    unit : str
        A string representing a unit of time (e.g., 'scale_02', 'scale_24').

    Returns
    -------
    float
        The duration in seconds corresponding to the input unit.
    """
    if unit == "scale_02":
        return 0.2
    if unit == "scale_04":
        return 0.4
    if unit == "scale_06":
        return 0.6
    if unit == "scale_2":
        return 2.0
    if unit == "scale_4":
        return 4.0
    if unit == "scale_6":
        return 6.0
    if unit == "scale_12":
        return 12.0
    if unit == "scale_24":
        return 24.0
    return 60.0


def segment_audio_files(
    df: pd.DataFrame,
    min_duration: float,
    output_dir: str,
    file_path_col: str,
    datetime_col: str,
) -> pd.DataFrame:
    """
    Segment audio files based on a minimum duration and create new DataFrame entries.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the audio file paths and timestamps.
    min_duration : float
        The minimum duration in seconds for each audio segment.
    output_dir : str
        Directory where the segmented audio files will be saved.
    file_path_col : str
        Column name for the audio file paths in the DataFrame.
    datetime_col : str
        Column name for the start time of the audio files in the DataFrame.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with new entries for each audio segment, including file paths and start/end times.

    Examples
    --------
    >>> from maui import samples, utils
    >>> df = samples.get_audio_sample(dataset="leec")
    >>> df["dt"] = pd.to_datetime(df["timestamp_init"]).dt.date
    >>> df = df.iloc[0:1]
    >>> segmented_df = utils.segment_audio_files(df, 0.2, './outputs', 'file_path', 'timestamp_init')
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    new_rows = []

    for _, row in df.iterrows():
        audio_path = row[file_path_col]
        try:
            with wave.open(audio_path, "rb") as wave_file:
                sample_rate = wave_file.getframerate()
                num_frames = wave_file.getnframes()
                audio_duration = num_frames / sample_rate
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue

        start_time = 0
        segment_number = 0
        initial_timestamp = row[datetime_col]

        while start_time < audio_duration:
            end_time = min(start_time + min_duration, audio_duration)
            segment_filename = f"{os.path.splitext(os.path.basename(audio_path))[0]}_segment_{segment_number}.wav"
            segment_path = os.path.join(output_dir, segment_filename)

            # Calculate frame positions
            start_frame = int(start_time * sample_rate)
            end_frame = int(end_time * sample_rate)
            num_frames_to_read = end_frame - start_frame

            try:
                with wave.open(audio_path, "rb") as wave_file:
                    wave_file.setpos(start_frame)
                    segment_frames = wave_file.readframes(num_frames_to_read)
                    with wave.open(segment_path, "wb") as segment_wave_file:
                        segment_wave_file.setnchannels(wave_file.getnchannels())
                        segment_wave_file.setsampwidth(wave_file.getsampwidth())
                        segment_wave_file.setframerate(sample_rate)
                        segment_wave_file.writeframes(segment_frames)
            except Exception as e:
                print(f"Error processing segment {segment_filename}: {e}")
                continue

            new_row = row.copy()
            new_row["segment_file_path"] = segment_path
            new_row["start_time"] = initial_timestamp + timedelta(seconds=start_time)
            new_row["end_time"] = initial_timestamp + timedelta(seconds=end_time)
            new_rows.append(new_row)

            start_time += min_duration
            segment_number += 1

    new_df = pd.DataFrame(new_rows)
    return new_df


def false_color_spectrogram_prepare_dataset(
    df,
    datetime_col: str,
    duration_col: str = None,
    file_path_col: str = None,
    output_dir: str = None,
    unit: str = "scale_60",
    calculate_acoustic_indices: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Prepare a dataset for generating false-color spectrograms, segmenting audio files,
    and calculating acoustic indices.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame containing the audio file paths and timestamps.
    datetime_col : str
        Column name for the start time of the audio files.
    duration_col : str, optional
        Column name for the duration of the audio files. If None, it will be calculated from file_path_col.
    file_path_col : str, optional
        Column name for the audio file paths. Required if duration_col is None.
    output_dir : str, optional
        Directory where segmented audio files and results will be stored.
    unit : str, optional
        Time unit for segmentation. Default is 'scale_60'.
    calculate_acoustic_indices : bool, optional
        If True, acoustic indices will be calculated for the segmented files.
    **kwargs : dict
        Additional parameters for calculating acoustic indices. The available kwargs are:

        - acoustic_indices_methods: list of str
            A list of methods used for calculating acoustic indices.

        - pre_calculation_method: callable
            A method to be applied before the calculation of acoustic indices.

        - parallel: bool
            Whether to perform the calculation of acoustic indices in parallel.

        - chunk_size: int, optional
            Size of the chunks of data to be processed in parallel. Default is 5.

        - temp_dir: str, optional
            Path to a temporary directory for storing intermediate results.

    Returns
    -------
    pandas.DataFrame
        A DataFrame containing the segmented audio files and, optionally, the calculated acoustic indices.

    Raises
    ------
    Exception
        If both duration_col and file_path_col are None, or if there are overlaps in the audio files, or if
        time gaps are detected between segments.

    Examples
    --------
    >>> from maui import samples, utils
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
    >>> 
    >>> df_temp = df.iloc[0:1]
    >>> segmented_df = utils.false_color_spectrogram_prepare_dataset(
    >>>     df_temp, 
    >>>     datetime_col = 'timestamp_init',
    >>>     duration_col = 'duration',
    >>>     file_path_col = 'file_path',
    >>>     indices = ['acoustic_complexity_index', 'spectral_activity', 'spectral_events'], 
    >>>     output_dir = './segmented_indices',
    >>>     store_audio_segments = True,
    >>>     unit = 'scale_02',
    >>>     acoustic_indices_methods = acoustic_indices_methods,
    >>>     pre_calculation_method = pre_calculation_method,
    >>>     temp_dir = os.path.abspath('./temp_ac_files/'),
    >>>     parallel = True
    >>> )
    """

    # 0.1. Verify if duration is already calculated or can be calculated
    if duration_col is None and file_path_col is None:
        raise Exception(
            "At least one of these arguments should not be None:"
            "duration_col, file_path_col"
        )
    available_units = [
        "scale_02",
        "scale_04",
        "scale_06",
        "scale_2",
        "scale_4",
        "scale_6",
        "scale_12",
        "scale_24",
        "scale_60",
    ]

    # 0.2. Verify unit is within the available units
    if unit not in available_units:
        raise Exception(
            f"""The unit {unit} is not available. """
            f"""The list of available units is: {available_units}"""
        )

    # 0.3. Verify if the duration of the audios is higher than the unit
    min_duration = _unit_conversion(unit)

    # 0.3.1. Calculate duration if not already calculated
    if duration_col is None:
        duration_col = "duration"
        df["duration"] = df[file_path_col].apply(_get_audio_duration)

    # 0.3.2. Raise exception if duration is smaller than unit
    all_durations_valid = df["duration"].ge(min_duration).all()
    if not all_durations_valid:
        raise Exception(
            f"""For {unit}, all the files must have duration greater or """
            f"""equal to {min_duration} seconds"""
        )

    # 0.4. Verify if there is overlap between files
    # 0.4.1. Calculate the end time for each audio
    df = df.copy()
    df["end_time"] = df[datetime_col] + pd.to_timedelta(df[duration_col], unit="s")

    # 0.4.2. Sort the DataFrame by start time
    df = df.sort_values(datetime_col).reset_index(drop=True)

    # 0.4.3. Check for overlaps
    overlap_exists = _check_overlaps(df, datetime_col)

    if overlap_exists:
        raise Exception(
            "To prepare the dataset correctly, the audios "
            "provided should not overlap"
        )

    # 0.5. Verify the existence of gaps in datetime
    time_gaps = _check_time_gaps(df, datetime_col)
    if time_gaps:
        gap_info = "; ".join(
            [f"Gap between {gap[0]} and {gap[1]}" for gap in time_gaps]
        )
        raise Exception(f"Time gaps found, remove them to continue: {gap_info}")

    # 1. Segment audio
    segmented_df = segment_audio_files(
        df, min_duration, output_dir, file_path_col, datetime_col
    )

    # 2. Calculate acoustic indices
    if calculate_acoustic_indices:
        segmented_df = acoustic_indices.calculate_acoustic_indices(
            segmented_df,
            "segment_file_path",
            kwargs["acoustic_indices_methods"],
            kwargs["pre_calculation_method"],
            parallel=kwargs["parallel"],
            chunk_size=kwargs["chunk_size"] if "chunk_size" in kwargs.keys() else 5,
            temp_dir=kwargs["temp_dir"],
        )

    return segmented_df


def get_blu_grn_palette():
    """
    Returns a colormap object that replicates the 'BluGrn' palette.

    This function creates a continuous colormap using a gradient from blue to green,
    which can be used to generate a range of colors in a plot.

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        A colormap object that can be used to map data values to colors.
    """
    cmap = mpl.colors.LinearSegmentedColormap.from_list("BluGrn", ["#0000FF", "#00FF00"])

    return cmap