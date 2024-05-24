"""
Module for handling audio file data and operations.

This module provides functionalities for loading audio datasets, extracting
information from audio files, and storing data efficiently. It includes utilities
for interacting with audio files and working with pandas DataFrames, facilitating
data manipulation and analysis workflows.

Dependencies:
- pandas: For DataFrame operations.
- pytest: For testing functionalities.
- os, zipfile: For file operations.
- maui.io: For loading audio datasets and extracting metadata.

Functions:
- sample_audio_files(tmpdir): Creates a temporary directory with sample audio files
  for testing purposes.
- test_get_audio_info_single_file(tmpdir): Tests for extracting audio information
  from a single audio file.
- test_get_audio_info_store_duration(tmpdir): Tests for extracting audio information
  with duration stored.
- test_get_audio_info_directory(tmpdir): Tests for extracting audio information
  from a directory of audio files.
- test_get_audio_info_directory_store_duration(tmpdir): Tests for extracting audio
  information from a directory of audio files with duration stored.
- test_get_audio_info_invalid_input(): Tests for handling invalid input when
  extracting audio information.
- sample_df(): Fixture for creating a sample DataFrame for testing.
- tmp_dir(tmpdir): Fixture for creating a temporary directory for testing.
- test_store_df_csv(sample_df_fixt, tmp_dir_fixt): Tests for storing a DataFrame
  in CSV format.
- test_store_df_pickle(sample_df_fixt, tmp_dir_fixt): Tests for storing a DataFrame
  in Pickle format.
- test_store_df_invalid_file_type(sample_df_fixt, tmp_dir_fixt): Tests for handling
  invalid file type when storing a DataFrame.

Note:
- This module is designed for testing audio file handling functionalities and
  ensuring data integrity and consistency.
"""

import os
import zipfile

import pandas as pd
import pytest

from maui import io


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

# Tests for get_audio_info method
# -------------------------------------------------------------------------
def test_get_audio_info_single_file(tmpdir):
    """
    Test for extracting audio information from a single file.

    Parameters
    ----------
    tmpdir : str
        The path to the temporary directory.
    """
    audio_file_path = sample_audio_files(tmpdir) + "/LEEC40__0__20170110_213100_br.wav"

    result_df = io.get_audio_info(str(audio_file_path), format_name="LEEC_FILE_FORMAT")

    assert not result_df.empty
    assert len(result_df) == 1
    assert "duration" not in result_df.columns

def test_get_audio_info_store_duration(tmpdir):
    """
    Test for extracting audio information with duration stored.

    Parameters
    ----------
    tmpdir : str
        The path to the temporary directory.
    """
    audio_file_path = sample_audio_files(tmpdir) + "/LEEC40__0__20170110_213100_br.wav"

    result_df = io.get_audio_info(
        str(audio_file_path), format_name="LEEC_FILE_FORMAT", store_duration=True
    )

    assert not result_df.empty
    assert len(result_df) == 1
    assert "duration" in result_df.columns

def test_get_audio_info_directory(tmpdir):
    """
    Test for extracting audio information from a directory.

    Parameters
    ----------
    tmpdir : str
        The path to the temporary directory.
    """
    audio_file_path = sample_audio_files(tmpdir)

    result_df = io.get_audio_info(audio_file_path, format_name="LEEC_FILE_FORMAT")

    assert not result_df.empty
    assert len(result_df) == 5
    assert "duration" not in result_df.columns

def test_get_audio_info_directory_store_duration(tmpdir):
    """
    Test for extracting audio information from a directory with duration stored.

    Parameters
    ----------
    tmpdir : str
        The path to the temporary directory.
    """
    audio_file_path = sample_audio_files(tmpdir)

    result_df = io.get_audio_info(
        audio_file_path, format_name="LEEC_FILE_FORMAT", store_duration=True
    )

    assert not result_df.empty
    assert len(result_df) == 5
    assert "duration" in result_df.columns

def test_get_audio_info_invalid_input():
    """Test for handling invalid input."""
    with pytest.raises(Exception):
        io.get_audio_info("non_existent_path", format_name="dummy_format")

# =========================================================================

# Tests for store_df method
# -------------------------------------------------------------------

@pytest.fixture(name="sample_df_fixt")
def sample_df():
    """
    Fixture for creating a sample DataFrame.

    Returns
    -------
    pandas.DataFrame
        A sample DataFrame.
    """
    data = {"A": [1, 2, 3], "B": ["a", "b", "c"]}
    return pd.DataFrame(data)

@pytest.fixture(name="tmp_dir_fixt")
def tmp_dir(tmpdir):
    """
    Fixture for creating a temporary directory.

    Parameters
    ----------
    tmpdir : str
        The path to the temporary directory.

    Returns
    -------
    str
        The path to the temporary directory.
    """
    return str(tmpdir)

# Tests for store_df function
def test_store_df_csv(sample_df_fixt, tmp_dir_fixt):
    """
    Test for storing a DataFrame in CSV format.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        A sample DataFrame.
    tmp_dir_fixt : str
        The path to the temporary directory.
    """
    file_name = "test_df"
    file_type = "csv"
    file_path = os.path.join(tmp_dir_fixt, file_name + ".csv")

    io.store_df(sample_df_fixt, file_type, tmp_dir_fixt, file_name)

    assert os.path.exists(file_path)

def test_store_df_pickle(sample_df_fixt, tmp_dir_fixt):
    """
    Test for storing a DataFrame in Pickle format.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        A sample DataFrame.
    tmp_dir_fixt : str
        The path to the temporary directory.
    """
    file_name = "test_df"
    file_type = "pickle"
    file_path = os.path.join(tmp_dir_fixt, file_name + ".pkl")

    io.store_df(sample_df_fixt, file_type, tmp_dir_fixt, file_name)

    assert os.path.exists(file_path)

def test_store_df_invalid_file_type(sample_df_fixt, tmp_dir_fixt):
    """
    Test for handling invalid file type.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        A sample DataFrame.
    tmp_dir_fixt : str
        The path to the temporary directory.
    """
    with pytest.raises(ValueError):
        io.store_df(sample_df_fixt, "invalid_file_type", tmp_dir_fixt, "test_df")
