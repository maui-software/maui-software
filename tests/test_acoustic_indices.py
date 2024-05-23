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

from maui import acoustic_indices, samples

def test_calculate_acoustic_indices(tmp_path):
    """
    Test `calculate_acoustic_indices` with serial execution.

    This test ensures that calling the `calculate_acoustic_indices` function
    with a valid dataset and executing it in serial mode correctly computes
    the specified acoustic indices.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory path provided by pytest.

    Assertions
    ----------
    AssertionError
        If the resulting DataFrame does not contain the expected columns.
    AssertionError
        If the number of rows in the resulting DataFrame does not match the
        input DataFrame.
    """
    tmp_dir = tmp_path / "maui_samples_leec"
    tmp_dir.mkdir()

    df = samples.get_audio_sample("leec", tmp_dir)
    df = df.head()

    indices_list = ["median_amplitude_envelope"]

    result_df = acoustic_indices.calculate_acoustic_indices(
        df, indices_list, store_df=False, parallel=False
    )

    assert "m" in result_df.columns
    assert len(result_df) == len(df)


def test_calculate_acoustic_indices_parallel(tmp_path):
    """
    Test `calculate_acoustic_indices` with parallel execution.

    This test ensures that calling the `calculate_acoustic_indices` function
    with a valid dataset and executing it in parallel mode correctly computes
    the specified acoustic indices.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory path provided by pytest.

    Assertions
    ----------
    AssertionError
        If the resulting DataFrame does not contain the expected columns.
    AssertionError
        If the number of rows in the resulting DataFrame does not match the
        input DataFrame.
    """
    tmp_dir = tmp_path / "maui_samples_leec"
    tmp_dir.mkdir()

    df = samples.get_audio_sample("leec", tmp_dir)
    df = df.head()

    indices_list = ["median_amplitude_envelope"]

    result_df = acoustic_indices.calculate_acoustic_indices(
        df, indices_list, store_df=False, parallel=True
    )

    assert "m" in result_df.columns
    assert len(result_df) == len(df)
