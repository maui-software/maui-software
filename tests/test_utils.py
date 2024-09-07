"""
Module for testing audio file processing methods such as segmentation and spectrogram preparation.

This module contains unit tests for the following public methods:
- `false_color_spectrogram_prepare_dataset`: Processes audio files to generate
  false-color spectrograms, while checking for overlaps, time gaps, and
  unit conversions.
- `segment_audio_files`: Segments audio files based on a minimum duration, ensuring
  valid file paths and handling audio duration calculations.

The module also includes helper functions to generate temporary audio files for testing.
"""

import zipfile
from pathlib import Path
from datetime import datetime

import pytest
import pandas as pd
from maui.utils import false_color_spectrogram_prepare_dataset, segment_audio_files
from maui import io


### Testing Private Methods via Public Methods ###


def sample_audio_files(tmpdir):
    """
    Create a temporary directory with sample audio files.

    Parameters
    ----------
    tmpdir : str
        The path to the temporary directory.

    Returns
    -------
    str
        The path to the directory containing the sample audio files.
    """
    audio_dir = tmpdir.mkdir("audio_files")

    sample_path = "./maui/data/audio_samples/leec_data_test_sample.zip"
    
    with zipfile.ZipFile(sample_path, "r") as zip_ref:
        zip_ref.extractall(audio_dir)

    return str(audio_dir)


# Test for overlap check (_check_overlaps tested through false_color_spectrogram_prepare_dataset)
def test_false_color_spectrogram_prepare_dataset_with_overlap():
    """
    Test the `false_color_spectrogram_prepare_dataset` function for detecting
    overlapping audio files.

    Raises
    ------
    Exception
        If the audio files overlap.
    """
    df = pd.DataFrame(
        {
            "start_time": [datetime(2023, 9, 1, 10, 0), datetime(2023, 9, 1, 10, 15)],
            "duration": [1800.0, 3600.0],  # First audio is 30 minutes, second is 1 hour
            "file_path": ["audio1.wav", "audio2.wav"],
        }
    )

    with pytest.raises(Exception, match="audios provided should not overlap"):

        false_color_spectrogram_prepare_dataset(
            df,
            datetime_col="start_time",
            duration_col="duration",
            file_path_col="file_path",
            output_dir="output",
        )


def test_false_color_spectrogram_prepare_dataset_with_time_gaps():
    """
    Test the `false_color_spectrogram_prepare_dataset` function for detecting
    time gaps between audio files.

    Raises
    ------
    Exception
        If time gaps are found between the audio files.
    """
    df = pd.DataFrame(
        {
            "start_time": [
                datetime(2023, 9, 1, 10, 0),
                datetime(2023, 9, 1, 12, 0),
            ],  # 2-hour gap
            "duration": [3600, 3600],  # Both audios are 1 hour long
            "file_path": ["audio1.wav", "audio2.wav"],
        }
    )

    with pytest.raises(Exception, match="Time gaps found, remove them to continue"):
        false_color_spectrogram_prepare_dataset(
            df,
            datetime_col="start_time",
            duration_col="duration",
            file_path_col="file_path",
            output_dir="output",
        )


# Test for unit conversion (_unit_conversion tested via false_color_spectrogram_prepare_dataset)
def test_false_color_spectrogram_prepare_dataset_unit_conversion():
    """
    Test the `false_color_spectrogram_prepare_dataset` function for unit =
    conversion when processing audio files.

    Raises
    ------
    Exception
        If the duration of the audio files is insufficient after unit conversion.
    """
    df = pd.DataFrame(
        {
            "start_time": [datetime(2023, 9, 1, 10, 0)],
            "duration": [7200],  # 2 hours long
            "file_path": ["audio1.wav"],
        }
    )

    with pytest.raises(
        Exception, match="all the files must have duration greater or equal to"
    ):
        false_color_spectrogram_prepare_dataset(
            df,
            datetime_col="start_time",
            file_path_col="file_path",
            output_dir="output",
            unit="scale_24",
        )


# Test for audio duration calculation
def test_false_color_spectrogram_prepare_dataset_audio_duration_calculation(tmpdir):
    """
    Test the automatic calculation of audio duration in the
    `false_color_spectrogram_prepare_dataset` function.

    Parameters
    ----------
    tmpdir : str
        The path to the temporary directory where audio files are located.

    Asserts
    -------
    AssertionError
        If the duration column is not added or if any duration is not calculated correctly.
    """
    audio_file_path = sample_audio_files(tmpdir) + "/LEEC40__0__20170110_213100_br.wav"

    df = io.get_audio_info(str(audio_file_path), format_name="LEEC_FILE_FORMAT")

    # Simulate the scenario where the duration is calculated if not provided
    output_df = false_color_spectrogram_prepare_dataset(
        df,
        datetime_col="timestamp_init",
        file_path_col="file_path",
        output_dir="output",
        calculate_acoustic_indices=False,
    )

    assert (
        "duration" in output_df.columns
    ), "Duration column should be added when not provided."
    assert output_df["duration"].notnull().all(), "All durations should be calculated."


### Direct Tests for Public Methods ###


# Test for segmenting audio files
def test_segment_audio_files(tmpdir):
    """
    Test the `segment_audio_files` function for correctly segmenting
    audio files based on a minimum duration.

    Parameters
    ----------
    tmpdir : str
        The path to the temporary directory where audio files are located.

    Asserts
    -------
    AssertionError
        If no segments are produced or if the end time is not after the start time for any segment.
    """
    audio_file_path = sample_audio_files(tmpdir) + "/LEEC40__0__20170110_213100_br.wav"

    df = io.get_audio_info(str(audio_file_path), format_name="LEEC_FILE_FORMAT")

    output_df = segment_audio_files(
        df,
        min_duration=1800,  # 30 minutes
        output_dir="output",
        file_path_col="file_path",
        datetime_col="timestamp_init",
    )

    assert len(output_df) > 0, "There should be at least one segment in the output."
    assert all(
        output_df["end_time"] > output_df["start_time"]
    ), "End time should be after start time."


# Test for segmenting audio with invalid file (public method testing exception handling)
def test_segment_audio_files_with_invalid_file():
    """
    Test the `segment_audio_files` function for handling invalid audio files.

    Asserts
    -------
    AssertionError
        If the output DataFrame is not empty when an invalid audio file is provided.
    """
    df = pd.DataFrame(
        {
            "start_time": [datetime(2023, 9, 1, 10, 0)],
            "file_path": ["invalid_audio.wav"],  # Simulate invalid file
        }
    )

    output_df = segment_audio_files(
        df,
        min_duration=1800,  # 30 minutes
        output_dir="output",
        file_path_col="file_path",
        datetime_col="start_time",
    )

    assert (
        output_df.empty
    ), "The output DataFrame should be empty when the audio file is invalid."


# Test for false_color_spectrogram_prepare_dataset with valid data and no overlaps or gaps
def test_false_color_spectrogram_prepare_dataset_no_overlap_no_gaps(tmpdir):
    """
    Test the `false_color_spectrogram_prepare_dataset` function with valid audio
    files that have no overlaps or time gaps.

    Parameters
    ----------
    tmpdir : str
        The path to the temporary directory where audio files are located.

    Asserts
    -------
    AssertionError
        If the output DataFrame is empty or does not contain segments.
    """
    audio_file_path = sample_audio_files(tmpdir)

    df = io.get_audio_info(str(audio_file_path), format_name="LEEC_FILE_FORMAT")
    df = df[df["timestamp_init"] < pd.Timestamp("2017-01-10 21:32:00")]

    output_df = false_color_spectrogram_prepare_dataset(
        df,
        datetime_col="timestamp_init",
        file_path_col="file_path",
        output_dir="output",
        calculate_acoustic_indices=False,
    )

    assert not output_df.empty, "The DataFrame should not be empty."
    assert len(output_df) > 0, "There should be segments in the output."


# Test for an empty DataFrame (edge case)
def test_false_color_spectrogram_no_files():
    """
    Test the `false_color_spectrogram_prepare_dataset` function with an empty DataFrame (edge case).

    Raises
    ------
    Exception
        If no valid files or parameters are provided.
    """
    df = pd.DataFrame()  # Empty DataFrame

    with pytest.raises(
        Exception, match="At least one of these arguments should not be None"
    ):
        false_color_spectrogram_prepare_dataset(df, datetime_col="start_time")


# Test for segment_audio_files with an empty DataFrame (edge case)
def test_segment_audio_files_no_files():
    """
    Test the `segment_audio_files` function with an empty DataFrame (edge case).

    Asserts
    -------
    AssertionError
        If the output DataFrame is not empty when no files are provided.
    """
    df = pd.DataFrame()  # Empty DataFrame
    output_df = segment_audio_files(
        df,
        min_duration=1800,
        output_dir="output",
        file_path_col="file_path",
        datetime_col="start_time",
    )

    assert (
        output_df.empty
    ), "The output DataFrame should be empty when there are no files to process."
