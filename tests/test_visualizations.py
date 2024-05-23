"""
Module for testing the visualizations module in the `maui` library.

This module contains tests for various visualization functions in the 
`maui` library, ensuring that these functions work correctly with sample 
data and produce the expected output.

Fixtures
--------
sample_df_fixt()
    Provides a sample DataFrame for testing.

sample_wav_file_fixt(tmp_path)
    Provides a sample .wav file for testing.

Tests
-----
test_indices_radar_plot(sample_df_fixt)
    Tests the `indices_radar_plot` function with valid parameters.

test_indices_radar_plot_invalid_agg_type(sample_df_fixt)
    Tests the `indices_radar_plot` function with an invalid aggregation 
    type.

test_indices_radar_plot_empty_indices(sample_df_fixt)
    Tests the `indices_radar_plot` function with empty indices.

test_indices_histogram_plot(sample_df_fixt)
    Tests the `indices_histogram_plot` function with valid parameters.

test_indices_histogram_plot_invalid_group_by(sample_df_fixt)
    Tests the `indices_histogram_plot` function with an invalid group 
    parameter.

test_indices_histogram_plot_empty_indices(sample_df_fixt)
    Tests the `indices_histogram_plot` function with empty indices.

test_indices_violin_plot(sample_df_fixt)
    Tests the `indices_violin_plot` function with valid parameters.

test_indices_violin_plot_invalid_group_by(sample_df_fixt)
    Tests the `indices_violin_plot` function with an invalid group 
    parameter.

test_indices_violin_plot_empty_indices(sample_df_fixt)
    Tests the `indices_violin_plot` function with empty indices.

test_spectrogram_plot(sample_wav_file_fixt)
    Tests the `spectrogram_plot` function with a valid .wav file.

test_spectrogram_plot_invalid_mode(sample_wav_file_fixt)
    Tests the `spectrogram_plot` function with an invalid mode.
"""

import wave
import pytest
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from maui import visualizations


@pytest.fixture(name="sample_df_fixt")
def sample_dataframe():
    """
    Provides a sample DataFrame for testing.

    Returns
    -------
    pd.DataFrame
        Sample DataFrame with predefined data.
    """
    data = {
        "index1": [0.1, 0.3, 0.5, 0.7, 0.9],
        "index2": [0.2, 0.4, 0.6, 0.8, 1.0],
        "group": ["A", "A", "B", "B", "C"],
    }
    return pd.DataFrame(data)


@pytest.fixture(name="sample_wav_file_fixt")
def sample_wav_file(tmp_path):
    """
    Provides a sample .wav file for testing.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory path provided by pytest.

    Returns
    -------
    pathlib.Path
        Path to the created .wav file.
    """
    file_path = tmp_path / "test.wav"

    # Parameters for the WAV file
    n_channels = 1  # Mono
    sampwidth = 2  # 2 bytes per sample
    framerate = 44100  # Sampling rate in Hz

    # Generate a dummy sine wave
    t = np.linspace(0, 1, framerate)
    data = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

    # Write the WAV file
    with wave.open(str(file_path), "wb") as wf:
        wf.setnchannels(n_channels)
        wf.setsampwidth(sampwidth)
        wf.setframerate(framerate)
        wf.writeframes(data.tobytes())

    return file_path


def test_indices_radar_plot(sample_df_fixt):
    """
    Tests the `indices_radar_plot` function with valid parameters.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `indices_radar_plot` function does not return the expected
        figure.
    """
    fig = visualizations.indices_radar_plot(
        sample_df_fixt, indices=["index1", "index2"], agg_type="mean", show_plot=False
    )
    assert isinstance(fig, go.Figure)
    assert "Radar Plot - Comparisson between indices" in fig.layout.title.text


def test_indices_radar_plot_invalid_agg_type(sample_df_fixt):
    """
    Tests the `indices_radar_plot` function with an invalid aggregation type.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `indices_radar_plot` function raises an AssertionError for
        invalid aggregation type.
    """
    with pytest.raises(AssertionError):
        visualizations.indices_radar_plot(
            sample_df_fixt,
            indices=["index1", "index2"],
            agg_type="invalid",
            show_plot=False,
        )


def test_indices_radar_plot_empty_indices(sample_df_fixt):
    """
    Tests the `indices_radar_plot` function with empty indices.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    IndexError
        If the `indices_radar_plot` function raises an IndexError for empty
        indices.
    """
    with pytest.raises(IndexError):
        visualizations.indices_radar_plot(
            sample_df_fixt, indices=[], agg_type="mean", show_plot=False
        )


def test_indices_histogram_plot(sample_df_fixt):
    """
    Tests the `indices_histogram_plot` function with valid parameters.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `indices_histogram_plot` function does not return the
        expected figure.
    """
    fig = visualizations.indices_histogram_plot(
        sample_df_fixt, indices=["index1", "index2"], show_plot=False
    )
    assert isinstance(fig, go.Figure)
    assert "Histogram Plot - Distribution of selected indices" in fig.layout.title.text


def test_indices_histogram_plot_invalid_group_by(sample_df_fixt):
    """
    Tests the `indices_histogram_plot` function with an invalid group
    parameter.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `indices_histogram_plot` function raises an AssertionError for
        invalid group parameter.
    """
    with pytest.raises(AssertionError):
        visualizations.indices_histogram_plot(
            sample_df_fixt, indices=["index1"], group_by="index3", show_plot=False
        )


def test_indices_histogram_plot_empty_indices(sample_df_fixt):
    """
    Tests the `indices_histogram_plot` function with empty indices.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    Exception
        If the `indices_histogram_plot` function raises an Exception for
        empty indices.
    """
    with pytest.raises(Exception):
        visualizations.indices_histogram_plot(sample_df_fixt, indices=[])


def test_indices_violin_plot(sample_df_fixt):
    """
    Tests the `indices_violin_plot` function with valid parameters.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `indices_violin_plot` function does not return the expected
        figure.
    """
    fig = visualizations.indices_violin_plot(
        sample_df_fixt, indices=["index1", "index2"], show_plot=False
    )
    assert isinstance(fig, go.Figure)
    assert "Violin Plot - Distribution of selected indices" in fig.layout.title.text


def test_indices_violin_plot_invalid_group_by(sample_df_fixt):
    """
    Tests the `indices_violin_plot` function with an invalid group parameter.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `indices_violin_plot` function raises an AssertionError for
        invalid group parameter.
    """
    with pytest.raises(AssertionError):
        visualizations.indices_violin_plot(
            sample_df_fixt, indices=["index1"], group_by="invalid", show_plot=False
        )


def test_indices_violin_plot_empty_indices(sample_df_fixt):
    """
    Tests the `indices_violin_plot` function with empty indices.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AttributeError
        If the `indices_violin_plot` function raises an AttributeError for
        empty indices.
    """
    with pytest.raises(AttributeError):
        visualizations.indices_violin_plot(sample_df_fixt, indices=[], show_plot=False)


def test_spectrogram_plot(sample_wav_file_fixt):
    """
    Test if spectrogram plot is generated correctly from a WAV file.

    Parameters
    ----------
    sample_wav_file_fixt : str
        Path to the sample WAV file.

    Assertions
    ----------
    AssertionError
        If the generated plot is not an instance of go.Figure
        or if the plot title does not contain expected text.
    """
    fig = visualizations.spectrogram_plot(
        file_path=str(sample_wav_file_fixt), mode="psd", show_plot=False
    )

    assert isinstance(fig, go.Figure)
    assert "Spectrogram generated from the file test.wav" in fig.layout.title.text


def test_spectrogram_plot_invalid_mode(sample_wav_file_fixt):
    """
    Test if an error is raised for invalid mode parameter.

    Parameters
    ----------
    sample_wav_file_fixt : str
        Path to the sample WAV file.

    Assertions
    ----------
    AssertionError
        If the call to spectrogram_plot with an invalid mode
        parameter does not raise AssertionError.
    """
    with pytest.raises(AssertionError):
        visualizations.spectrogram_plot(
            file_path=str(sample_wav_file_fixt), mode="invalid", show_plot=False
        )
