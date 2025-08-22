"""
Module for testing the EDA functions from the `maui` library.

This module contains tests for various EDA (Exploratory Data Analysis) 
functions in the `maui` library, ensuring that these functions work 
correctly with sample data and produce the expected output.

Functions
---------
sample_df()
    Fixture to provide a sample DataFrame for testing.

test_card_summary(sample_df)
    Test the `card_summary` function with valid and invalid categories.

test_heatmap_analysis(sample_df)
    Test the `heatmap_analysis` function with the sample DataFrame.

test_stacked_bar_analysis(sample_df)
    Test the `stacked_bar_analysis` function with the sample DataFrame.

test_duration_analysis(sample_df)
    Test the `duration_analysis` function with the sample DataFrame.

test_daily_distribution_analysis(sample_df)
    Test the `daily_distribution_analysis` function with the sample 
    DataFrame.

test_duration_distribution(sample_df)
    Test the `duration_distribution` function with the sample DataFrame.
"""

import pytest
import pandas as pd
import plotly.graph_objs as go
from maui import eda
from maui.eda.eda import CategoryLimitError


@pytest.fixture(name="sample_df_fixt")
def sample_df():
    """
    Provide a sample DataFrame for testing.

    Returns
    -------
    pd.DataFrame
        Sample DataFrame with predefined data.
    """
    data = {
        "file_path": ["file1", "file2", "file3", "file4"],
        "dt": ["2023-01-01", "2023-01-02", "2023-01-01", "2023-01-02"],
        "duration": [100, 200, 300, 400],
        "landscape": ["urban", "rural", "urban", "rural"],
        "environment": ["indoor", "outdoor", "indoor", "outdoor"],
    }
    return pd.DataFrame(data)


def test_card_summary(sample_df_fixt):
    """
    Test the `card_summary` function with valid and invalid categories.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `card_summary` function does not return the expected
        dictionary and figure.
    CategoryLimitError
        If more than two categories are passed to the `card_summary`
        function.
    """
    categories = ["landscape", "environment"]
    card_dict, fig = eda.card_summary(sample_df_fixt, categories, show_plot=False)

    assert isinstance(card_dict, dict)
    assert isinstance(fig, go.Figure)
    assert card_dict["n_samples"] == 4
    assert card_dict["distinct_days"] == 2
    assert card_dict["total_time_duration"] == 1000 / 60
    assert card_dict["mean_time_duration"] == 250 / 60
    assert card_dict["landscape"] == 2
    assert card_dict["environment"] == 2

    with pytest.raises(CategoryLimitError) as excinfo:
        eda.card_summary(
            sample_df_fixt, ["landscape", "environment", "extra_category"], show_plot=False
        )
    assert str(excinfo.value) == "At most two categories should be selected."


def test_heatmap_analysis(sample_df_fixt):
    """
    Test the `heatmap_analysis` function with the sample DataFrame.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `heatmap_analysis` function does not return the expected
        DataFrame and figure.
    """
    df_group, fig = eda.heatmap_analysis(
        sample_df_fixt, "landscape", "environment", show_plot=False
    )

    assert isinstance(df_group, pd.DataFrame)
    assert isinstance(fig, go.Figure)
    assert df_group.shape == (4, 3)
    assert "landscape" in df_group.columns
    assert "environment" in df_group.columns
    assert "count" in df_group.columns


def test_stacked_bar_analysis(sample_df_fixt):
    """
    Test the `stacked_bar_analysis` function with the sample DataFrame.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `stacked_bar_analysis` function does not return the expected
        figure.
    """
    fig = eda.stacked_bar_analysis(sample_df_fixt, "landscape", "environment", show_plot=False)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == (
        "Amount of samples by landscape and segmented by environment"
    )


def test_duration_analysis(sample_df_fixt):
    """
    Test the `duration_analysis` function with the sample DataFrame.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `duration_analysis` function does not return the expected
        figure.
    """
    fig = eda.duration_analysis(sample_df_fixt, "landscape", "duration", show_plot=False)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Duration distribution by landscape"


def test_daily_distribution_analysis(sample_df_fixt):
    """
    Test the `daily_distribution_analysis` function with the sample
    DataFrame.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `daily_distribution_analysis` function does not return the
        expected figure.
    """
    fig = eda.daily_distribution_analysis(sample_df_fixt, "dt", "landscape", show_plot=False)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == ("Amount of samples by Day and landscape")


def test_duration_distribution(sample_df_fixt):
    """
    Test the `duration_distribution` function with the sample DataFrame.

    Parameters
    ----------
    sample_df_fixt : pd.DataFrame
        Sample DataFrame provided by the fixture.

    Assertions
    ----------
    AssertionError
        If the `duration_distribution` function does not return the expected
        figure.
    """
    fig = eda.duration_distribution(sample_df_fixt, show_plot=False)

    assert isinstance(fig, go.Figure)
    assert fig.layout.title.text == "Duration distribution"
