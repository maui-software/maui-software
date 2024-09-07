"""
Module for testing the `acoustic_indices` module from the `maui` 
library.

This module contains tests for the `calculate_acoustic_indices` function, 
which computes various acoustic indices from audio samples. The tests verify 
the correctness of the function's output when executed in both serial and 
parallel modes.

Functions
---------
test_calculate_acoustic_indices(tmp_path)
    Test `calculate_acoustic_indices` with serial execution.

test_calculate_acoustic_indices_parallel(tmp_path)
    Test `calculate_acoustic_indices` with parallel execution.
"""

import pytest
import pandas as pd
import numpy as np
from unittest import mock

from maad import sound

from maui import acoustic_indices, samples


def pre_calc_method(s, fs):
    """
    Prepare the sound data for acoustic index calculation.

    Parameters
    ----------
    s : np.ndarray
        The sound data array.
    fs : int
        The sampling frequency of the sound data.

    Returns
    -------
    dict
        A dictionary containing the sound array as value for key 's'.
    """
    return {'s': s}  # Pass the sound array as the pre-calculation variable

def acoustic_method(pre_calc):
    """
    Compute an example acoustic index from the pre-calculated data.

    Parameters
    ----------
    pre_calc : dict
        A dictionary containing the pre-calculated data. Expected to have
        the key 's' with the sound array.

    Returns
    -------
    dict
        A dictionary containing the computed acoustic index under key 'index1'.
    """
    return {'index1': np.mean(pre_calc['s'])}  # Example index: mean of the sound array

@pytest.fixture
def sample_dataframe():
    """
    Fixture to create a sample DataFrame for testing.

    Returns
    -------
    pd.DataFrame
        A DataFrame with columns 'file_path' and 'metadata' containing sample audio file paths
        and metadata strings.
    """
    data = {
        'file_path': ['audio1.wav'],
        'metadata': ['[1, 2, 3]']  # Strings to be converted to lists
    }
    return pd.DataFrame(data)

@mock.patch("os.makedirs")
@mock.patch("pandas.DataFrame.to_csv")
@mock.patch("pandas.read_csv")  # Mock read_csv
@mock.patch("maad.sound.load")
@mock.patch("os.remove")  # Mock os.remove to avoid FileNotFoundError
def test_calculate_acoustic_indices(mock_remove, mock_load, mock_read_csv, mock_to_csv, mock_makedirs, sample_dataframe):
    """
    Test the `calculate_acoustic_indices` function to ensure it correctly calculates acoustic indices
    for all files in the DataFrame when executed in serial mode.

    Parameters
    ----------
    mock_remove : MagicMock
        Mock object for `os.remove` to avoid file deletion errors.
    mock_load : MagicMock
        Mock object for `maad.sound.load` to simulate audio data loading.
    mock_read_csv : MagicMock
        Mock object for `pandas.read_csv` to simulate reading CSV files.
    mock_to_csv : MagicMock
        Mock object for `pandas.DataFrame.to_csv` to avoid actual file writing.
    mock_makedirs : MagicMock
        Mock object for `os.makedirs` to simulate directory creation.
    sample_dataframe : pd.DataFrame
        A sample DataFrame with file paths and metadata for testing.

    Returns
    -------
    None
    """
    # Mock sound.load to return dummy audio data
    mock_load.return_value = (np.array([1, 2, 3]), 44100)
    data = {
        'file_path': ['audio1.wav'],
        'metadata': ['[1, 2, 3]'],
        'index1': [np.mean([1, 2, 3])]  # Strings to be converted to lists
    }

    # Mock read_csv to return a dummy DataFrame when it is called
    mock_read_csv.return_value = pd.DataFrame(data)

    # Call the calculate_acoustic_indices function
    result_df = acoustic_indices.calculate_acoustic_indices(
        sample_dataframe,
        file_path_col='file_path',
        acoustic_indices_methods=[acoustic_method],
        pre_calculation_method=pre_calc_method,
        parallel=False,
        chunk_size=1,
        temp_dir='./temp_dir'
    )

    print('result_df: ', result_df)
    print('sample_dataframe: ', sample_dataframe)

    # Ensure the directory is created
    mock_makedirs.assert_called_with('./temp_dir', exist_ok=True)

    # Verify if 'index1' is in the resulting DataFrame
    assert 'index1' in result_df.columns
    assert len(result_df) == len(sample_dataframe)

    # Ensure the result contains valid calculated indices
    assert result_df['index1'].iloc[0] == np.mean([1, 2, 3])



    # Verify that the 'file_path' column is preserved and has correct values
    assert all(result_df['file_path'] == sample_dataframe['file_path'])

    # Verify that os.remove was called with the correct paths
    expected_temp_files = ['./temp_dir/temp_0.csv']  # Adjust this if your temp files have different names
    mock_remove.assert_called_with(expected_temp_files[0])


@mock.patch("os.makedirs")
@mock.patch("pandas.DataFrame.to_csv")
@mock.patch("pandas.read_csv")  # Mock read_csv
@mock.patch("maad.sound.load")
@mock.patch("os.remove")
def test_calculate_acoustic_indices_parallel(mock_remove, mock_load, mock_read_csv, mock_to_csv, mock_makedirs, sample_dataframe):
    """
    Test the `calculate_acoustic_indices` function in parallel mode to ensure that it works correctly.

    Parameters
    ----------
    mock_remove : MagicMock
        Mock object for `os.remove` to avoid file deletion errors.
    mock_load : MagicMock
        Mock object for `maad.sound.load` to simulate audio data loading.
    mock_read_csv : MagicMock
        Mock object for `pandas.read_csv` to simulate reading CSV files.
    mock_to_csv : MagicMock
        Mock object for `pandas.DataFrame.to_csv` to avoid actual file writing.
    mock_makedirs : MagicMock
        Mock object for `os.makedirs` to simulate directory creation.
    sample_dataframe : pd.DataFrame
        A sample DataFrame with file paths and metadata for testing.

    Returns
    -------
    None
    """
    mock_load.return_value = (np.array([1, 2, 3]), 44100)
    data = {
        'file_path': ['audio1.wav'],
        'metadata': ['[1, 2, 3]'],
        'index1': [np.mean([1, 2, 3])]  # Strings to be converted to lists
    }

    # Mock read_csv to return a dummy DataFrame when it is called
    mock_read_csv.return_value = pd.DataFrame(data)
    
    result_df = acoustic_indices.calculate_acoustic_indices(
        sample_dataframe,
        file_path_col='file_path',
        acoustic_indices_methods=[acoustic_method],
        pre_calculation_method=pre_calc_method,
        parallel=True,  # Test in parallel mode
        chunk_size=1,
        temp_dir='./temp_dir'
    )

    # Ensure the directory is created
    mock_makedirs.assert_called_with('./temp_dir', exist_ok=True)

    # Verify if 'index1' is in the resulting DataFrame
    assert 'index1' in result_df.columns
    assert len(result_df) == len(sample_dataframe)

    # Ensure the result contains valid calculated indices
    assert result_df['index1'].iloc[0] == np.mean([1, 2, 3])