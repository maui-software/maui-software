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
import pandas as pd
import pytest
from maui import samples


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
