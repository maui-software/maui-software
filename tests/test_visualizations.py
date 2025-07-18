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
from plotly.graph_objs import Figure

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


# ------------ Test visualizations.diel_plot -------------------------------------
def test_diel_plot_time_parsing():
    """
    Test that the 'time' column is correctly parsed into a 24-hour format.

    This function verifies that the diel_plot function correctly converts
    time strings in different formats (e.g., '9am', '10:30 PM') to the
    'HH:MM' 24-hour format. It also checks if the returned object is a
    Plotly figure.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the 'time' column is not correctly parsed or the returned object
        is not a Plotly figure.
    """
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "time": ["9am", "10:30 PM"],  # These times should be parsed
            "length": [10, 15],
            "group": ["1", "2"],
        }
    )

    fig = visualizations.diel_plot(
        df, "date", "time", "length", 15, "group", show_plot=False
    )

    # Check that the returned object is a Plotly figure
    assert isinstance(fig, Figure)

    # Ensure that time is correctly parsed into 24-hour format
    assert df["time"].tolist() == ["09:00", "22:30"]


def test_diel_plot_time_truncation():
    """
    Test that the 'time' column is truncated based on the time_bin_size.

    This function verifies that the diel_plot function truncates the time
    values to the nearest bin, as defined by the time_bin_size parameter.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the 'time' column is not correctly truncated based on the
        time_bin_size.
    """
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01"],
            "time": [
                "09:07",
                "10:44",
            ],  # These times should be truncated based on time_bin_size
            "length": [10, 15],
            "group": ["1", "2"],
        }
    )

    _ = visualizations.diel_plot(
        df, "date", "time", "length", 20, "group", show_plot=False
    )

    # Check that the time column has been truncated to the nearest bin
    assert df["time"].tolist() == ["09:00", "10:40"]


def test_diel_plot_aggregation():
    """
    Test that aggregation by mean is correctly applied in diel_plot.

    This function checks that the diel_plot function correctly aggregates
    values by the 'mean' when the agg_type parameter is set to 'mean'.
    It ensures that the result is not based on counts.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the aggregation is not correctly performed or the plot is
        counting samples instead.
    """
    df = pd.DataFrame(
        {
            "date": ["2024-01-01", "2024-01-01", "2024-01-02"],
            "time": ["09:00", "09:00", "10:00"],
            "length": [10, 20, 30],
            "group": [1, 2, 3],
        }
    )

    # Check aggregation by mean
    fig = visualizations.diel_plot(
        df, "date", "time", "length", 40, "group", agg_type="mean", show_plot=False
    )

    # The mean of group values for 2024-01-01 09:00 should be calculated
    assert (
        "Number of samples" not in fig.data[0].colorbar.title.text
    )  # Verify it's not counting


def test_diel_plot_invalid_time_bin_size():
    """
    Test that an AttributeError is raised for invalid time_bin_size.

    This function checks that the diel_plot function raises an AttributeError
    when an invalid time_bin_size (not between 1 and 60) is provided.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AttributeError
        If time_bin_size is not an integer between 1 and 60.
    """
    df = pd.DataFrame(
        {"date": ["2024-01-01"], "time": ["09:00"], "length": [5], "group": ["1"]}
    )

    with pytest.raises(
        AttributeError, match="time_bin_size must be an integer between 1 and 60"
    ):
        _ = visualizations.diel_plot(df, "date", "time", "length", 65, "group")


def test_diel_plot_none_agg_type():
    """
    Test that an AttributeError is raised when agg_type is None.

    This function checks that the diel_plot function raises an AttributeError
    when agg_type is None, which is not a valid value.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    AttributeError
        If agg_type is set to None.
    """
    df = pd.DataFrame(
        {"date": ["2024-01-01"], "time": ["09:00"], "length": [5], "group": [1]}
    )

    with pytest.raises(AttributeError, match="agg_type should not be None"):
        _ = visualizations.diel_plot(df, "date", "time", "length", 30, "group")


def test_diel_plot_warning_duration_greater_than_time_bin_size():
    """
    Test that a UserWarning is raised when duration exceeds time_bin_size.

    This function checks that the diel_plot function raises a UserWarning
    when a row has a duration greater than the time_bin_size parameter.

    Parameters
    ----------
    None

    Returns
    -------
    None

    Raises
    ------
    UserWarning
        If rows have a duration greater than the time_bin_size.
    """
    df = pd.DataFrame(
        {
            "date": ["2024-01-01"],
            "time": ["09:00"],
            "length": [3000],  # Greater than time_bin_size
            "group": ["1"],
        }
    )

    with pytest.warns(
        UserWarning, match="rows have a duration greater than the time_bin_size"
    ):
        _ = visualizations.diel_plot(
            df, "date", "time", "length", 15, "group", show_plot=False
        )


# ------------ Test visualizations.polar_bar_plot -------------------------------------
def test_polar_bar_plot_valid_input():
    """
    Test the `polar_bar_plot` function with valid input.

    This test verifies that the function generates a valid plotly figure
    when provided with a valid DataFrame. Specifically, it checks that the
    returned object is a `plotly.graph_objects.Figure`.

    Parameters
    ----------
    None

    Raises
    ------
    AssertionError
        If the returned object is not a plotly Figure.
    """
    # Create a valid DataFrame for testing
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=366, freq="D"),
            "category": ["A", "B", "C"] * 122,
        }
    )

    # Test the function with valid input
    fig = visualizations.polar_bar_plot(
        df, "date", "category", percent=False, show_plot=False
    )

    # Check if the returned object is a plotly figure
    assert isinstance(fig, go.Figure)


def test_missing_column():
    """
    Test the `polar_bar_plot` function with valid input.

    This test verifies that the function generates a valid plotly figure
    when provided with a valid DataFrame. Specifically, it checks that the
    returned object is a `plotly.graph_objects.Figure`.

    Parameters
    ----------
    None

    Raises
    ------
    AssertionError
        If the returned object is not a plotly Figure.

    """
    # Create a DataFrame missing the required columns
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=366, freq="D"),
            "wrong_category": ["A", "B", "C"] * 122,
        }
    )

    # Expect an AssertionError due to the missing category column
    with pytest.raises(AssertionError):
        visualizations.polar_bar_plot(
            df, "date", "category", percent=False, show_plot=False
        )


def test_invalid_category_type():
    """
    Test the `polar_bar_plot` function with invalid category types.

    This test checks that the function raises an `AttributeError` when the
    `categories_col` contains continuous data instead of discrete categories.

    Parameters
    ----------
    None

    Raises
    ------
    AttributeError
        If the `categories_col` contains continuous data instead of discrete categories.
    """
    # Create a DataFrame with invalid category type (continuous instead of discrete)
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=365, freq="D"),
            "category": np.linspace(
                0, 1, 365
            ),  # Continuous values instead of categorical
        }
    )

    # Expect an AttributeError due to continuous data in the category column
    with pytest.raises(AttributeError):
        visualizations.polar_bar_plot(
            df, "date", "category", percent=False, show_plot=False
        )


def test_invalid_dates_warning():
    """
    Test the `polar_bar_plot` function with invalid date values.

    This test checks that the function raises a `UserWarning` when invalid
    date values are present in the DataFrame. Rows with invalid dates should
    be ignored in the plot.

    Parameters
    ----------
    None

    Raises
    ------
    UserWarning
        If invalid dates are found in the `date_time_col`.
    """
    # Create a DataFrame with invalid dates
    df = pd.DataFrame(
        {
            "date": ["2023-01-01", "invalid_date", "2023-01-03"],
            "category": ["A", "B", "C"],
        }
    )

    # Expect a warning due to the invalid date
    with pytest.warns(UserWarning, match="Warning: 1 rows have invalid dates"):
        visualizations.polar_bar_plot(
            df, "date", "category", percent=False, show_plot=False
        )


def test_percentage_mode():
    """
    Test the `polar_bar_plot` function with percentage mode enabled.

    This test verifies that the function returns a valid plotly figure when
    the `percent` parameter is set to True. The plot should display data as
    percentages rather than raw counts.

    Parameters
    ----------
    None

    Raises
    ------
    AssertionError
        If the returned object is not a plotly Figure.
    """
    # Create a valid DataFrame for testing
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=366, freq="D"),
            "category": ["A", "B", "C"] * 122,
        }
    )

    # Test the function with percent=True
    fig = visualizations.polar_bar_plot(
        df, "date", "category", percent=True, show_plot=False
    )

    # Check if the returned object is a plotly figure
    assert isinstance(fig, go.Figure)


def test_figure_custom_dimensions():
    """
    Test the `polar_bar_plot` function with custom figure dimensions.

    This test verifies that the function correctly applies custom height and
    width dimensions to the plot layout when passed as keyword arguments.

    Parameters
    ----------
    None

    Raises
    ------
    AssertionError
        If the figure layout height or width does not match the expected values.
    """
    # Create a valid DataFrame for testing
    df = pd.DataFrame(
        {
            "date": pd.date_range(start="2023-01-01", periods=366, freq="D"),
            "category": ["A", "B", "C"] * 122,
        }
    )

    # Test with custom figure dimensions
    fig = visualizations.polar_bar_plot(
        df, "date", "category", percent=False, show_plot=False, height=500, width=700
    )

    # Check if the layout has the correct height and width
    assert fig.layout.height == 500
    assert fig.layout.width == 700


# ------------ Test visualizations.parallel_coordinates_plot -------------------------------------
def test_returns_figure(sample_df_fixt):
    """
    Test that the parallel_coordinates_plot function returns a Figure.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        A fixture providing a sample DataFrame to use as plot input.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the returned object is not an instance of Figure.

    Notes
    -----
    Asserts that the figure returned by the plot function is of the correct type.
    """
    fig = visualizations.parallel_coordinates_plot(
        sample_df_fixt, ["index1", "index2"], color_col="group", show_plot=False
    )
    assert isinstance(fig, Figure)


def test_axes_labels(sample_df_fixt):
    """
    Test that dimension labels in the parallel coordinates plot match input columns.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        A fixture with data columns used for plotting.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the plot labels do not match the specified input indices.

    Notes
    -----
    Ensures that the correct labels are displayed on the plot axes.
    """
    fig = visualizations.parallel_coordinates_plot(
        sample_df_fixt, ["index1", "index2"], color_col="group", show_plot=False
    )
    labels = [dim["label"] for dim in fig.data[0]["dimensions"]]
    assert set(labels) == {"index1", "index2"}


def test_color_array_length(sample_df_fixt):
    """
    Test that the color array length matches the number of DataFrame rows.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        Input data for plotting.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the color array length is not equal to the number of rows in the DataFrame.

    Notes
    -----
    Verifies correct mapping between data and coloring.
    """
    fig = visualizations.parallel_coordinates_plot(
        sample_df_fixt, ["index1", "index2"], color_col="group", show_plot=False
    )
    color_arr = fig.data[0]["line"]["color"]
    assert len(color_arr) == len(sample_df_fixt)


def test_numeric_coloring(sample_df_fixt):
    """
    Test support for numeric coloring in the parallel coordinates plot.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        Input data, extended with a numeric coloring column.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the returned object is not a Figure or labels are incorrect.

    Notes
    -----
    Confirms that numeric features can be used for plot coloring and that labels remain accurate.
    """
    sample_df_fixt["numeric_group"] = [1, 2, 1, 2, 3]
    fig = visualizations.parallel_coordinates_plot(
        sample_df_fixt, ["index1", "index2"], color_col="numeric_group", show_plot=False
    )
    assert isinstance(fig, Figure)
    labels = [dim["label"] for dim in fig.data[0]["dimensions"]]
    assert set(labels) == {"index1", "index2"}


def test_empty_indices_raises(sample_df_fixt):
    """
    Test that providing an empty list of indices raises IndexError.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        Input data for plotting.

    Returns
    -------
    None

    Raises
    ------
    IndexError
        If indices list is empty.

    Notes
    -----
    Checks input validation for the expected minimum number of indices.
    """
    with pytest.raises(IndexError):
        visualizations.parallel_coordinates_plot(
            sample_df_fixt, [], color_col="group", show_plot=False
        )


def test_one_index_raises(sample_df_fixt):
    """
    Test that providing a single index for parallel coordinates raises IndexError.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        Input data for plotting.

    Returns
    -------
    None

    Raises
    ------
    IndexError
        If only one index is provided.

    Notes
    -----
    Verifies enforcement of the minimum required number of plotting dimensions.
    """
    with pytest.raises(IndexError):
        visualizations.parallel_coordinates_plot(
            sample_df_fixt, ["index1"], color_col="group", show_plot=False
        )


def test_index_not_in_df_raises(sample_df_fixt):
    """
    Test that referencing a non-existent index column raises AssertionError.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        Input data for plotting.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If an index is not present in the DataFrame columns.

    Notes
    -----
    Ensures robust handling of invalid column references.
    """
    with pytest.raises(AssertionError):
        visualizations.parallel_coordinates_plot(
            sample_df_fixt, ["index2", "not_found"], color_col="group", show_plot=False
        )


def test_color_col_missing_raises(sample_df_fixt):
    """
    Test that specifying a missing color column raises AssertionError.

    Parameters
    ----------
    sample_df_fixt : pandas.DataFrame
        Input DataFrame for plotting.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If the specified color_col does not exist in the DataFrame.

    Notes
    -----
    Ensures proper validation and error reporting for the coloring column argument.
    """
    with pytest.raises(AssertionError):
        visualizations.parallel_coordinates_plot(
            sample_df_fixt,
            ["index1", "index2"],
            color_col="missing_col",
            show_plot=False,
        )
