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
from unittest import mock

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


def test_false_color_spectrogram_plot_valid_input():
    """
    Test the `false_color_spectrogram_plot` function with valid input.

    This test creates a DataFrame with 1D arrays for each index column,
    runs the `false_color_spectrogram_plot` function, and verifies
    that the output spectrogram has the expected shape and dtype.

    Assertions
    ----------
    - The shape of the resulting spectrogram should be (10, 100, 3).
    - The spectrogram values should be normalized to uint8 dtype, in the range [0, 255].

    Raises
    ------
    AssertionError
        If the shape or dtype of the spectrogram is not as expected.
    """
    # Create a sample dataframe
    data = {
        "timestamp": pd.date_range("2023-01-01", periods=100, freq="s"),
        "index1": [np.random.rand(10) for _ in range(100)],
        "index2": [np.random.rand(10) for _ in range(100)],
        "index3": [np.random.rand(10) for _ in range(100)],
    }
    df = pd.DataFrame(data)

    print(df)

    indices = ["index1", "index2", "index3"]

    # Run the function with valid input
    spectrogram = visualizations.false_color_spectrogram_plot(
        df, "timestamp", indices, display=False
    )

    # Assertions
    assert spectrogram.shape == (
        10,
        100,
        3,
    ), "The spectrogram should have a shape of (100, 10, 3)."
    assert (
        spectrogram.dtype == np.uint8
    ), "The spectrogram values should be in the range of [0, 255]."


def test_false_color_spectrogram_plot_empty_indices():
    """
    Test the `false_color_spectrogram_plot` function with an empty indices list.

    This test verifies that the function raises an `IndexError` when called
    with an empty list of acoustic indices.

    Raises
    ------
    IndexError
        If the indices list is empty.
    """
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="s"),
            "index1": np.random.rand(100),
        }
    )

    with pytest.raises(IndexError):
        visualizations.false_color_spectrogram_plot(df, "timestamp", [], display=False)


def test_false_color_spectrogram_plot_invalid_index():
    """
    Test the `false_color_spectrogram_plot` function with an invalid index.

    This test verifies that the function raises an `AssertionError` when the
    specified index is not found in the DataFrame columns.

    Raises
    ------
    AssertionError
        If the specified index is not present in the DataFrame.
    """
    df = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-01", periods=100, freq="s"),
            "index1": np.random.rand(100),
        }
    )

    with pytest.raises(AssertionError):
        visualizations.false_color_spectrogram_plot(
            df, "timestamp", ["invalid_index"], display=False
        )


def test_false_color_spectrogram_plot_invalid_unit():
    """
    Test the `false_color_spectrogram_plot` function with an invalid unit.

    This test checks that the function raises an `Exception` when an invalid
    unit is passed for truncating timestamps.

    Raises
    ------
    Exception
        If the unit is not one of the accepted time units.
    """
    data = {
        "timestamp": pd.date_range("2023-01-01", periods=100, freq="s"),
        "index1": [np.random.rand(10) for _ in range(100)],
        "index2": [np.random.rand(10) for _ in range(100)],
        "index3": [np.random.rand(10) for _ in range(100)],
    }
    df = pd.DataFrame(data)

    with pytest.raises(Exception):
        visualizations.false_color_spectrogram_plot(
            df,
            "timestamp",
            ["index1", "index2", "index3"],
            unit="invalid_unit",
            display=False,
        )


@mock.patch("plotly.graph_objects.Figure.show")
def test_display_false_color_spectrogram_plot(mock_show):
    """
    Test the display functionality of `false_color_spectrogram_plot`.

    This test creates a DataFrame with 1D arrays for the index columns,
    runs the `false_color_spectrogram_plot` function with `display=True`,
    and checks if the plot is displayed using Plotly.

    Parameters
    ----------
    mock_show : mock.Mock
        Mock object for Plotly's `Figure.show` function to prevent
        actual display during the test.

    Assertions
    ----------
    - Verifies that the `show` function was called exactly once.
    """
    # Create a sample dataframe
    data = {
        "timestamp": pd.date_range("2023-01-01", periods=100, freq="s"),
        "index1": [np.random.rand(10) for _ in range(100)],
        "index2": [np.random.rand(10) for _ in range(100)],
        "index3": [np.random.rand(10) for _ in range(100)],
    }
    df = pd.DataFrame(data)

    indices = ["index1", "index2", "index3"]

    # Test the display functionality with valid data
    visualizations.false_color_spectrogram_plot(
        df, "timestamp", indices, display=True, fig_size={"width": 1000, "height": 500}
    )

    # Assert that the show function was called to display the plot
    mock_show.assert_called_once()
