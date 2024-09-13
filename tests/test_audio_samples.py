"""
Module for testing the `get_audio_sample` function from the `maui` library.

This module contains tests for the `get_audio_sample` function, which is used
to retrieve audio samples from specified datasets. The tests ensure that the
function behaves correctly with both valid and invalid dataset names.

Functions
---------
test_get_audio_sample_invalid_dataset()
    Test `get_audio_sample` with an invalid dataset.

test_get_audio_sample_valid_dataset_leec(tmp_path)
    Test `get_audio_sample` with a valid dataset (LEEC).
"""

import os
import sys
from unittest.mock import patch, MagicMock
from pathlib import Path

import pandas as pd
import pytest

from maui import samples

sys.modules["tqdm"] = MagicMock()


def test_get_audio_sample_invalid_dataset():
    """
    Test `get_audio_sample` with an invalid dataset.

    This test ensures that calling the `get_audio_sample` function with an
    invalid dataset name raises an Exception.

    Raises
    ------
    Exception
        If the dataset name is invalid.
    """
    with pytest.raises(Exception):
        samples.get_audio_sample(dataset="invalid_dataset")


def test_get_audio_sample_valid_dataset_leec(tmp_path):
    """
    Test `get_audio_sample` with a valid dataset (LEEC).

    This test ensures that calling the `get_audio_sample` function with a
    valid dataset name ("leec") correctly retrieves audio samples and
    stores them in the specified directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory path provided by pytest.

    Assertions
    ----------
    AssertionError
        If the returned DataFrame is not an instance of pd.DataFrame.
    AssertionError
        If the returned DataFrame is empty.
    AssertionError
        If the number of files in the directory is not equal to 120.
    """
    tmp_dir = tmp_path / "maui_samples_leec"
    tmp_dir.mkdir()

    # Call the function with a valid dataset
    df = samples.get_audio_sample("leec", tmp_dir)

    # Assertions
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert len(os.listdir(tmp_dir)) == 120


# Test for get_xc_data
@patch("requests.get")
@patch("urllib.request.urlretrieve")
@patch("os.path.exists", return_value=False)
@patch("os.makedirs")
def test_get_xc_data_success(
    mock_makedirs, mock_exists, mock_urlretrieve, mock_requests_get
):
    """
    Test the `get_xc_data` function for a successful API request and file download.

    This test checks the case where the API request is successful, and the files
    are downloaded and saved to the specified directory. The test mocks the API
    response and the file download function to simulate a successful response
    from the Xeno-canto API.

    Parameters
    ----------
    mock_makedirs : MagicMock
        Mock for the `os.makedirs` function to simulate directory creation.
    mock_exists : MagicMock
        Mock for the `os.path.exists` function to simulate the file existence check.
    mock_urlretrieve : MagicMock
        Mock for the `urllib.request.urlretrieve` function to simulate file download.
    mock_requests_get : MagicMock
        Mock for the `requests.get` function to simulate the API request.

    Raises
    ------
    AssertionError
        If the DataFrame does not contain the expected columns or values.

    Notes
    -----
    - The test mocks the response from the Xeno-canto API and simulates the download
    of audio files.
    - After calling `get_xc_data`, it checks that the DataFrame contains the expected
    columns and values.
    """
    # Mock the API response for _get_xc_dataset
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "recordings": [
            {
                "id": "1",
                "file-name": "sample.mp3",
                "file": "http://example.com/sample.mp3",
            }
        ]
    }
    mock_requests_get.return_value = mock_response

    # Mock the file download in _download_xc_files
    mock_urlretrieve.return_value = ("/path/to/sample.mp3", None)

    q = {"gen": "Pica", "cnt": "brazil"}
    extract_path = Path("./test_files")

    # Call the actual method without mocking the private methods
    df = samples.get_xc_data(q, extract_path)

    assert isinstance(df, pd.DataFrame)
    assert "local_file_path" in df.columns
    assert "file_downloaded" in df.columns
    assert df["file_downloaded"].iloc[0] == 1
    assert df["local_file_path"].iloc[0] == os.path.join(extract_path, "sample.mp3")


@patch("requests.get")
@patch("os.path.exists", return_value=True)
@patch("os.makedirs")
def test_get_xc_data_file_exists(mock_makedirs, mock_exists, mock_requests_get):
    """
    Test the `get_xc_data` function when the files already exist in the target directory.

    This test simulates the case where the file to be downloaded already exists in the
    target directory.
    The test mocks the API response and checks that the file download is skipped if
    the file is already present.

    Parameters
    ----------
    mock_makedirs : MagicMock
        Mock for the `os.makedirs` function to simulate directory creation.
    mock_exists : MagicMock
        Mock for the `os.path.exists` function to simulate the file existence check.
    mock_requests_get : MagicMock
        Mock for the `requests.get` function to simulate the API request.

    Raises
    ------
    AssertionError
        If the DataFrame does not contain the expected columns or values.

    Notes
    -----
    - The test mocks the Xeno-canto API response to provide sample recording data.
    - The test checks that the file is not downloaded again if it already exists in
    the specified directory.
    """
    # Mock the API response for _get_xc_dataset
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "recordings": [
            {
                "id": "1",
                "file-name": "sample.mp3",
                "file": "http://example.com/sample.mp3",
            }
        ]
    }
    mock_requests_get.return_value = mock_response

    q = {"gen": "Pica", "cnt": "brazil"}
    extract_path = Path("./test_files")

    # Call the actual method without mocking the private methods
    df = samples.get_xc_data(q, extract_path)
    print(df)

    assert isinstance(df, pd.DataFrame)
    assert "local_file_path" in df.columns
    assert "file_downloaded" in df.columns
    assert df["file_downloaded"].iloc[0] == 1
    assert df["local_file_path"].iloc[0] == os.path.join(extract_path, "sample.mp3")


@patch("requests.get")
def test_get_xc_data_no_recordings(mock_requests_get):
    """
    Test the `get_xc_data` function when no recordings are found by the API.

    This test simulates the scenario where the API returns a valid response but
    contains no recordings.
    It ensures that the function returns an empty DataFrame when there are no
    recordings.

    Parameters
    ----------
    mock_requests_get : MagicMock
        Mock for the `requests.get` function to simulate the API request.

    Raises
    ------
    AssertionError
        If the returned DataFrame is not empty as expected.

    Notes
    -----
    - The test simulates an API response with an empty list of recordings.
    - It verifies that the `get_xc_data` function returns an empty DataFrame when
    no recordings are found.
    """
    # Mock the API response for _get_xc_dataset with no recordings
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"recordings": []}
    mock_requests_get.return_value = mock_response

    q = {"gen": "Pica", "cnt": "brazil"}
    extract_path = "./test_files"

    # Call the actual method
    df = samples.get_xc_data(q, extract_path)

    assert isinstance(df, pd.DataFrame)
    assert df.empty


def test_get_xc_data_invalid_query():
    """
    Test the `get_xc_data` function with invalid query parameters.

    This test ensures that the function raises a `ValueError` when the query
    dictionary contains invalid keys.
    It verifies the function's ability to validate query parameters against
    an allowed set.

    Raises
    ------
    ValueError
        If the query contains unexpected keys.

    Notes
    -----
    - The test provides an invalid query dictionary and checks if the
    function raises a `ValueError`.
    - It ensures that the `get_xc_data` function only accepts valid query
    parameters.
    """
    q = {"invalid_key": "Pica"}
    extract_path = "./test_files"

    with pytest.raises(ValueError, match="Unexpected keys found"):
        samples.get_xc_data(q, extract_path)
